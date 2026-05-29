import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Deque, List
from collections import deque
import time
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging

logger = logging.getLogger("CTBRTools")


# ==============================================
# 1. 数据类定义
# ==============================================

@dataclass
class ObservationData:
    """封装从飞控收到的所有状态数据"""
    time_boot_ms: int = 0

    # ATTITUDE (ID 30)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # LOCAL_POSITION_NED (ID 32)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # ACTUATOR_OUTPUT_STATUS (ID 375)
    motors: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class ActionData:
    """封装发送给飞控的控制指令"""
    time_sent_local: float = 0.0  # 本地发送时间戳
    time_boot_ms_est: int = 0     # 估算的飞控时间戳
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.0

@dataclass
class EpisodeStatus:
    episode_id: int = -1
    step_id: int = -1
    phase: str = "idle"          # takeoff / hover / collect / recover / done
    done: bool = False
    done_reason: str = ""
    crashed: bool = False


@dataclass
class RuntimeStatus:
    armed: bool = False
    base_mode: int = 0
    custom_mode: int = 0
    flight_mode: str = ""
    failsafe: bool = False
    last_status_text: str = ""

@dataclass
class SyncedData:
    """接收数据与发送指令对齐拼接"""
    obs: ObservationData = field(default_factory=ObservationData)
    last_action: Optional[ActionData] = None
    received_wall_time: float = field(default_factory=time.time)
    episode_status: EpisodeStatus = field(default_factory=EpisodeStatus)
    runtime_status: RuntimeStatus = field(default_factory=RuntimeStatus)


# ==============================================
# 2. 同步数据日志记录器
# ==============================================

class SyncedDataLogger:
    """
    专门用于记录对齐后的 SyncedData：Queue 缓冲 + 批量 Parquet 写入。

    [MOD] 原实现每次 flush 都重新创建 ParquetWriter，可能覆盖旧批次。
          现在改为一个 logger 生命周期内维护同一个 ParquetWriter，
          stop() 时统一 flush 并 close。
    """

    def __init__(
        self,
        log_dir: str = "./log_folder",
        batch_size: int = 100,
        filename: Optional[str] = None,
    ):
        self._log_dir = log_dir
        self._batch_size = batch_size
        os.makedirs(self._log_dir, exist_ok=True)

        self._queue = queue.Queue(maxsize=10000)
        self._buffer: List[Dict] = []

        self._running = False
        self._writer_thread: Optional[threading.Thread] = None

        # [MOD] 持久 writer，避免反复打开同一个 parquet 文件导致覆盖。
        self._writer: Optional[pq.ParquetWriter] = None
        self._writer_lock = threading.Lock()
        self._dropped_count = 0

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._filepath = os.path.join(self._log_dir, f"trajectory_{timestamp}.parquet")
        else:
            if not filename.endswith(".parquet"):
                filename += ".parquet"
            self._filepath = os.path.join(self._log_dir, filename)

    @property
    def filepath(self) -> str:
        """[MOD] 便于外部打印/检查日志路径。"""
        return self._filepath

    def start(self):
        """启动日志写入线程"""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        logger.info(f"[SyncedDataLogger] 已启动，日志文件: {self._filepath}")

    def stop(self):
        """停止日志写入线程，强制刷新剩余数据并关闭 writer"""
        # [MOD] 即使 _running 已经 False，也尝试 flush/close，保证 cleanup 幂等。
        self._running = False

        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)

        self._drain_queue_to_buffer(max_items=None)
        self._flush_buffer(force=True)
        self._close_writer()

        if self._dropped_count > 0:
            logger.warning(f"[SyncedDataLogger] 队列满导致丢弃 {self._dropped_count} 条数据")

        logger.info("[SyncedDataLogger] 已停止，数据已保存")

    def log_synced_data(self, data: SyncedData):
        """
        非阻塞写入：将对齐数据推入队列。
        这个函数会被 DroneDataSync 调用。
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self._dropped_count += 1

    def _write_loop(self):
        """后台线程循环：从队列取数据并批量写入"""
        while self._running or not self._queue.empty():
            try:
                self._drain_queue_to_buffer(
                    max_items=self._batch_size if self._running else None
                )

                if len(self._buffer) >= self._batch_size or (not self._running and self._buffer):
                    self._flush_buffer(force=not self._running)

                if self._running and self._queue.empty():
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"[SyncedDataLogger] 写入错误: {e}", exc_info=True)
                time.sleep(0.05)

    def _drain_queue_to_buffer(self, max_items: Optional[int] = None):
        """[MOD] 将 queue 中的数据取出到内存 buffer。"""
        count = 0

        while max_items is None or count < max_items:
            try:
                synced_data = self._queue.get(timeout=0.03 if self._running else 0.0)
            except queue.Empty:
                break

            self._buffer.append(self._synced_data_to_dict(synced_data))
            count += 1

    def _flush_buffer(self, force: bool = False):
        """将缓冲区写入 Parquet"""
        if not self._buffer:
            return

        if len(self._buffer) < self._batch_size and not force:
            return

        try:
            df = pd.DataFrame(self._buffer)
            table = pa.Table.from_pandas(df, preserve_index=False)

            with self._writer_lock:
                if self._writer is None:
                    self._writer = pq.ParquetWriter(self._filepath, table.schema)

                self._writer.write_table(table)

            written = len(self._buffer)
            self._buffer = []
            logger.debug(f"[SyncedDataLogger] 成功写入 {written} 条数据到 {self._filepath}")

        except Exception as e:
            # [MOD] 写入失败时不清空 buffer，避免静默丢数据。
            logger.error(f"[SyncedDataLogger] Flush 错误: {e}", exc_info=True)

    def _close_writer(self):
        """[MOD] 安全关闭 ParquetWriter。"""
        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.close()
                finally:
                    self._writer = None

    def _synced_data_to_dict(self, data: SyncedData) -> Dict:
        """将 SyncedData 转换为字典"""
        d = {}
        ep = data.episode_status
        rt = data.runtime_status

        d["episode_id"] = ep.episode_id
        d["step_id"] = ep.step_id
        d["phase"] = ep.phase
        d["done"] = ep.done
        d["done_reason"] = ep.done_reason
        d["crashed"] = ep.crashed

        d["armed"] = rt.armed
        d["base_mode"] = rt.base_mode
        d["custom_mode"] = rt.custom_mode
        d["flight_mode"] = rt.flight_mode
        d["failsafe"] = rt.failsafe
        d["last_status_text"] = rt.last_status_text

        # 1. 现实时间
        d["wall_time"] = data.received_wall_time
        d["wall_time_str"] = datetime.fromtimestamp(data.received_wall_time).strftime("%Y-%m-%d %H:%M:%S.%f")

        # 2. 飞控时间
        d["px4_time_boot_ms"] = data.obs.time_boot_ms

        # 3. 姿态
        d["roll"] = data.obs.roll
        d["pitch"] = data.obs.pitch
        d["yaw"] = data.obs.yaw
        d["rollspeed"] = data.obs.rollspeed
        d["pitchspeed"] = data.obs.pitchspeed
        d["yawspeed"] = data.obs.yawspeed

        # 4. 位置
        d["x"] = data.obs.x
        d["y"] = data.obs.y
        d["z"] = data.obs.z
        d["vx"] = data.obs.vx
        d["vy"] = data.obs.vy
        d["vz"] = data.obs.vz

        # 5. 电机
        d["motor_1"] = data.obs.motors[0] if len(data.obs.motors) > 0 else 0.0
        d["motor_2"] = data.obs.motors[1] if len(data.obs.motors) > 1 else 0.0
        d["motor_3"] = data.obs.motors[2] if len(data.obs.motors) > 2 else 0.0
        d["motor_4"] = data.obs.motors[3] if len(data.obs.motors) > 3 else 0.0

        # 6. 上条指令估算时间
        if data.last_action:
            d["last_action_px4_time_est"] = data.last_action.time_boot_ms_est
            # 7. 控制指令
            d["cmd_body_roll_rate"] = data.last_action.body_roll_rate
            d["cmd_body_pitch_rate"] = data.last_action.body_pitch_rate
            d["cmd_body_yaw_rate"] = data.last_action.body_yaw_rate
            d["cmd_thrust"] = data.last_action.thrust
        else:
            d["last_action_px4_time_est"] = 0
            d["cmd_body_roll_rate"] = 0.0
            d["cmd_body_pitch_rate"] = 0.0
            d["cmd_body_yaw_rate"] = 0.0
            d["cmd_thrust"] = 0.0

        return d


# ==============================================
# 3. 核心同步类
# ==============================================

class DroneDataSync:
    """
    飞控观测与控制动作的同步器。

    [MOD] 原实现存在两个问题：
    1. SimTimeKeeper 只有在 ATTITUDE/LOCAL_POSITION/ACTUATOR 三帧齐全时才被 notify，
       如果 actuator 消息缺失或不同步，仿真时间等待可能卡住。
    2. 使用 _time_lock 和 _lock 两把锁，存在锁顺序反转风险。
       现在统一使用一把 RLock + Condition。
    """

    def __init__(
        self,
        use_condition: bool = True,
        action_history_len: int = 10,
        sync_window_ms: int = 50,
        logger: Optional[SyncedDataLogger] = None,
    ):
        self._use_condition = use_condition
        self._sync_window_ms = sync_window_ms
        self._logger = logger

        # [MOD] 统一锁，避免 _time_lock 与 _lock 锁顺序反转。
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        self._latest_px4_time_est: int = 0
        self._last_update_wall_time: float = 0.0
        self._has_new_data = False
        self._latest_obs: ObservationData = ObservationData()
        self._action_history: Deque[ActionData] = deque(maxlen=action_history_len)
        self._frame_cache = {
            "ATTITUDE": None,
            "LOCAL_POSITION_NED": None,
            "ACTUATOR_OUTPUT_STATUS": None,
        }
        self._episode_status = EpisodeStatus()
        self._runtime_status = RuntimeStatus()

    def _extract_msg_time_ms(self, msg) -> int:
        """[MOD] 统一提取 MAVLink 消息时间戳，单位 ms。"""
        if hasattr(msg, "time_boot_ms"):
            return int(msg.time_boot_ms)
        if hasattr(msg, "time_usec"):
            return int(msg.time_usec // 1000)
        return 0

    def on_new_observation(self, msg):
        """
        接收飞控消息。

        [MOD] 任意带时间戳的观测消息都会推进 _latest_px4_time_est 并 notify_all，
              这样 SimTimeKeeper 不再依赖完整同步帧。
        """
        msg_type = msg.get_type()
        current_px4_time = self._extract_msg_time_ms(msg)

        with self._cv:
            if current_px4_time > 0:
                # 时间戳可能偶尔乱序；这里取 max，保证仿真时间单调不回退。
                self._latest_px4_time_est = max(self._latest_px4_time_est, current_px4_time)
                self._cv.notify_all()

            if not self._use_condition:
                return

            if msg_type in self._frame_cache:
                self._frame_cache[msg_type] = msg

            self._try_emit_synced_frame_locked()

    def on_new_action(self, action: ActionData):
        """时间戳填充和动作记录"""
        with self._cv:
            if action.time_boot_ms_est == 0:
                action.time_boot_ms_est = self._latest_px4_time_est

            if self._use_condition:
                self._action_history.append(action)
                self._cv.notify_all()

    def wait_for_synced_data(self, timeout: float = None) -> Optional[SyncedData]:
        """阻塞等待新数据，并自动对齐"""
        if not self._use_condition:
            return None

        with self._cv:
            signaled = self._cv.wait_for(lambda: self._has_new_data, timeout=timeout)
            if not signaled:
                return None

            self._has_new_data = False
            obs = ObservationData(**self._latest_obs.__dict__)
            action = self._select_action_for_time_locked(obs.time_boot_ms)

            return SyncedData(
                obs=obs,
                last_action=ActionData(**action.__dict__) if action else None,
                received_wall_time=time.time(),
                episode_status=self._snapshot_episode_status_locked(),
                runtime_status=self._snapshot_runtime_status_locked(),
            )

    def get_latest_px4_time_ms(self) -> int:
        """获取当前最新的飞控时间戳"""
        with self._lock:
            return self._latest_px4_time_est

    def get_latest_observation(self) -> ObservationData:
        """[MOD] 获取最新观测的快照，便于 RL 主循环直接取 obs。"""
        with self._lock:
            return ObservationData(**self._latest_obs.__dict__)

    def _select_action_for_time_locked(self, obs_time_ms: int) -> Optional[ActionData]:
        """
        [MOD] 选择与 obs_time_ms 对齐的动作。
        优先选择最后一条 time_boot_ms_est <= obs_time_ms 的动作；
        如果没有，则退化为最新动作。
        """
        if not self._action_history:
            return None

        selected = None
        for action in self._action_history:
            if action.time_boot_ms_est <= obs_time_ms:
                selected = action
            else:
                break

        return selected if selected is not None else self._action_history[-1]

    def _try_emit_synced_frame_locked(self):
        """
        消息集齐，推送完整信息。
        调用时必须已经持有 self._cv / self._lock。
        """
        cache = self._frame_cache
        if not (cache["ATTITUDE"] and cache["LOCAL_POSITION_NED"] and cache["ACTUATOR_OUTPUT_STATUS"]):
            return

        t_att = self._extract_msg_time_ms(cache["ATTITUDE"])
        t_pos = self._extract_msg_time_ms(cache["LOCAL_POSITION_NED"])
        t_act = self._extract_msg_time_ms(cache["ACTUATOR_OUTPUT_STATUS"])

        times = [t_att, t_pos, t_act]
        max_t = max(times)
        min_t = min(times)

        if max_t - min_t > self._sync_window_ms:
            # [MOD] 原代码直接 return，会导致旧帧永久卡在 cache 中。
            # 这里丢弃最旧消息，等待下一帧重新对齐。
            for key, msg in list(cache.items()):
                if msg is not None and self._extract_msg_time_ms(msg) == min_t:
                    cache[key] = None
            return

        self._update_obs_with_msg(self._latest_obs, cache["ATTITUDE"], "ATTITUDE")
        self._update_obs_with_msg(self._latest_obs, cache["LOCAL_POSITION_NED"], "LOCAL_POSITION_NED")
        self._update_obs_with_msg(self._latest_obs, cache["ACTUATOR_OUTPUT_STATUS"], "ACTUATOR_OUTPUT_STATUS")

        # 以本帧中较新的时间戳作为观测时间，避免被不同消息覆盖成较旧时间。
        self._latest_obs.time_boot_ms = max_t
        self._latest_px4_time_est = max(self._latest_px4_time_est, max_t)

        self._has_new_data = True
        self._last_update_wall_time = time.time()

        if self._logger:
            action = self._select_action_for_time_locked(self._latest_obs.time_boot_ms)
            synced_data = SyncedData(
                obs=ObservationData(**self._latest_obs.__dict__),
                last_action=ActionData(**action.__dict__) if action else None,
                received_wall_time=time.time(),
                episode_status=self._snapshot_episode_status_locked(),
                runtime_status=self._snapshot_runtime_status_locked(),
            )
            self._logger.log_synced_data(synced_data)

        self._cv.notify_all()
        self._frame_cache = {k: None for k in self._frame_cache}

    def _update_obs_with_msg(self, obs: ObservationData, msg, msg_type: str):
        if msg_type == "ATTITUDE":
            obs.roll = msg.roll
            obs.pitch = msg.pitch
            obs.yaw = msg.yaw
            obs.rollspeed = msg.rollspeed
            obs.pitchspeed = msg.pitchspeed
            obs.yawspeed = msg.yawspeed

        elif msg_type == "LOCAL_POSITION_NED":
            obs.x = msg.x
            obs.y = msg.y
            obs.z = msg.z
            obs.vx = msg.vx
            obs.vy = msg.vy
            obs.vz = msg.vz

        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            obs.motors = list(msg.actuator[:4])

    def set_episode(
        self,
        episode_id: int,
        phase: str = "collect",
        step_id: int = 0,
    ):
        with self._cv:
            self._episode_status.episode_id = episode_id
            self._episode_status.phase = phase
            self._episode_status.step_id = step_id
            self._episode_status.done = False
            self._episode_status.done_reason = ""
            self._episode_status.crashed = False
            self._cv.notify_all()


    def set_step(self, step_id: int):
        with self._cv:
            self._episode_status.step_id = step_id


    def set_phase(self, phase: str):
        with self._cv:
            self._episode_status.phase = phase


    def mark_done(
        self,
        done_reason: str = "normal",
        crashed: bool = False,
    ):
        with self._cv:
            self._episode_status.done = True
            self._episode_status.done_reason = done_reason
            self._episode_status.crashed = crashed
            self._episode_status.phase = "done"
            self._cv.notify_all()


    def clear_done(self):
        with self._cv:
            self._episode_status.done = False
            self._episode_status.done_reason = ""
            self._episode_status.crashed = False
            self._cv.notify_all()


    def update_runtime_status(
        self,
        armed: bool = None,
        base_mode: int = None,
        custom_mode: int = None,
        flight_mode: str = None,
        failsafe: bool = None,
        last_status_text: str = None,
    ):
        with self._cv:
            if armed is not None:
                self._runtime_status.armed = bool(armed)
            if base_mode is not None:
                self._runtime_status.base_mode = int(base_mode)
            if custom_mode is not None:
                self._runtime_status.custom_mode = int(custom_mode)
            if flight_mode is not None:
                self._runtime_status.flight_mode = str(flight_mode)
            if failsafe is not None:
                self._runtime_status.failsafe = bool(failsafe)
            if last_status_text is not None:
                self._runtime_status.last_status_text = str(last_status_text)
            self._cv.notify_all()


    def _snapshot_episode_status_locked(self) -> EpisodeStatus:
        return EpisodeStatus(**self._episode_status.__dict__)


    def _snapshot_runtime_status_locked(self) -> RuntimeStatus:
        return RuntimeStatus(**self._runtime_status.__dict__)


# ==============================================
# 4. 仿真时间对齐
# ==============================================

class SimTimeKeeper:
    """仿真时间对齐工具类。替代 time.sleep() 和 asyncio.sleep()，让程序按照仿真时间流速运行。"""

    def __init__(self, data_sync: DroneDataSync):
        if not hasattr(data_sync, "_cv"):
            raise RuntimeError("SimTimeKeeper 需要 DroneDataSync 开启 use_condition=True!")
        self._data_sync = data_sync
        self._cv = data_sync._cv
        self._lock = data_sync._lock

    def now_ms(self) -> int:
        """获取当前仿真时间 (毫秒)"""
        return self._data_sync.get_latest_px4_time_ms()

    def wait(self, seconds: float, timeout: float = None) -> bool:
        """
        阻塞等待，直到仿真时间流逝指定秒数。
        替代 time.sleep(seconds)。
        :param seconds: 仿真世界中需要等待的时长
        :param timeout: 现实世界中的超时时间，None为无限等待
        :return: True 表示等待完成，False 表示现实超时
        """
        start_sim_ms = self._wait_for_first_tick(timeout=timeout)
        if start_sim_ms == 0:
            return False

        target_sim_ms = start_sim_ms + int(seconds * 1000)
        return self._wait_until(target_sim_ms, timeout)

    def wait_until(self, target_time_ms: int, timeout: float = None) -> bool:
        """阻塞等待，直到仿真时间到达某个特定的时间戳"""
        return self._wait_until(target_time_ms, timeout)

    async def wait_async(self, seconds: float, timeout: float = None) -> bool:
        """
        异步等待，用于 async def 函数中。
        替代 await asyncio.sleep(seconds)
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.wait, seconds, timeout)

    def _wait_for_first_tick(self, timeout: float = None) -> int:
        """等待直到接收到第一帧带时间戳的数据"""
        start_wall = time.time()

        with self._cv:
            while self._data_sync._latest_px4_time_est == 0:
                if timeout is not None and time.time() - start_wall >= timeout:
                    return 0
                self._cv.wait(0.1)

            return self._data_sync._latest_px4_time_est

    def _wait_until(self, target_ms: int, timeout: float = None) -> bool:
        """核心等待逻辑"""
        start_wall_time = time.time()

        with self._cv:
            while True:
                current_ms = self._data_sync._latest_px4_time_est

                if current_ms >= target_ms:
                    return True

                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_wall_time
                    if elapsed >= timeout:
                        return False
                    remaining_timeout = timeout - elapsed

                wait_timeout = min(remaining_timeout, 0.1) if remaining_timeout is not None else 0.1

                try:
                    self._cv.wait_for(
                        lambda: self._data_sync._latest_px4_time_est >= target_ms,
                        timeout=wait_timeout,
                    )
                except KeyboardInterrupt:
                    logger.warning("SimTimeKeeper 等待被中断")
                    return False


# ==============================================
# 5. 控制命令参数类
# ==============================================

@dataclass
class ControlParams:
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.55
