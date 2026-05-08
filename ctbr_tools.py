import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Deque, List
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
class SyncedData:
    """接收数据与发送指令对齐拼接"""
    obs: ObservationData = field(default_factory=ObservationData)
    last_action: Optional[ActionData] = None
    received_wall_time: float = field(default_factory=time.time)

# ==============================================
# 2. 同步数据日志记录器
# ==============================================

class SyncedDataLogger:
    """
    专门用于记录对齐后的 SyncedData,Queue 缓冲 + 批量 Parquet 写入"""
    def __init__(self, log_dir: str = "./log_folder", batch_size: int = 100):
        self._log_dir = log_dir
        self._batch_size = batch_size
        os.makedirs(self._log_dir, exist_ok=True)
        self._queue = queue.Queue(maxsize=10000)
        self._buffer: List[Dict] = []
        
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None
        
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self._log_dir, f"trajectory_{timestamp}.parquet")

    def start(self):
        """启动日志写入线程"""
        if self._running:
            return
        self._running = True
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        logger.info(f"[SyncedDataLogger] 已启动，日志文件: {self._filepath}")

    def stop(self):
        """停止日志写入线程，强制刷新剩余数据"""
        if not self._running:
            return
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
        self._flush_buffer(force=True)
        logger.info(f"[SyncedDataLogger] 已停止，数据已保存")

    def log_synced_data(self, data: SyncedData):
        """
        非阻塞写入：将对齐数据推入队列
        这个函数会被 DroneDataSync 调用
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            pass

    def _write_loop(self):
        """后台线程循环：从队列取数据并批量写入"""
        while self._running or not self._queue.empty():
            try:
                count = 0
                while count < self._batch_size and (self._running or not self._queue.empty()):
                    try:
                        synced_data = self._queue.get(timeout=0.05)
                        self._buffer.append(self._synced_data_to_dict(synced_data))
                        count += 1
                    except queue.Empty:
                        break
                
                if len(self._buffer) >= self._batch_size or (not self._running and self._buffer):
                    self._flush_buffer()

            except Exception as e:
                logger.error(f"[SyncedDataLogger] 写入错误: {e}")

    def _flush_buffer(self, force: bool = False):
        """将缓冲区写入 Parquet"""
        if not self._buffer:
            return
        
        if len(self._buffer) < self._batch_size and not force:
            return

        try:
            df = pd.DataFrame(self._buffer)
            
            if not os.path.exists(self._filepath):
                table = pa.Table.from_pandas(df)
                pq.write_table(table, self._filepath)
            else:
                old_df = pd.read_parquet(self._filepath)
                combined_df = pd.concat([old_df, df], ignore_index=True)
                table = pa.Table.from_pandas(combined_df)
                pq.write_table(table, self._filepath)
            
            self._buffer = []
            
        except Exception as e:
            logger.error(f"[SyncedDataLogger] Flush 错误: {e}")

    def _synced_data_to_dict(self, data: SyncedData) -> Dict:
        """将 SyncedData 转换为字典"""
        d = {}
        
        # 1. 现实时间
        d['wall_time'] = data.received_wall_time
        d['wall_time_str'] = datetime.fromtimestamp(data.received_wall_time).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # 2. 飞控时间
        d['px4_time_boot_ms'] = data.obs.time_boot_ms
        
        # 3. 姿态
        d['roll'] = data.obs.roll
        d['pitch'] = data.obs.pitch
        d['yaw'] = data.obs.yaw
        d['rollspeed'] = data.obs.rollspeed
        d['pitchspeed'] = data.obs.pitchspeed
        d['yawspeed'] = data.obs.yawspeed
        
        # 4. 位置
        d['x'] = data.obs.x
        d['y'] = data.obs.y
        d['z'] = data.obs.z
        d['vx'] = data.obs.vx
        d['vy'] = data.obs.vy
        d['vz'] = data.obs.vz
        
        # 5. 电机
        d['motor_1'] = data.obs.motors[0] if len(data.obs.motors) > 0 else 0.0
        d['motor_2'] = data.obs.motors[1] if len(data.obs.motors) > 1 else 0.0
        d['motor_3'] = data.obs.motors[2] if len(data.obs.motors) > 2 else 0.0
        d['motor_4'] = data.obs.motors[3] if len(data.obs.motors) > 3 else 0.0
        
        # 6. 上条指令估算时间
        if data.last_action:
            d['last_action_px4_time_est'] = data.last_action.time_boot_ms_est
            # 7. 控制指令
            d['cmd_body_roll_rate'] = data.last_action.body_roll_rate
            d['cmd_body_pitch_rate'] = data.last_action.body_pitch_rate
            d['cmd_body_yaw_rate'] = data.last_action.body_yaw_rate
            d['cmd_thrust'] = data.last_action.thrust
        else:
            d['last_action_px4_time_est'] = 0
            d['cmd_body_roll_rate'] = 0.0
            d['cmd_body_pitch_rate'] = 0.0
            d['cmd_body_yaw_rate'] = 0.0
            d['cmd_thrust'] = 0.0
            
        return d

# ==============================================
# 3. 核心同步类
# ==============================================

class DroneDataSync:
    def __init__(self, 
                 use_condition: bool = True, 
                 action_history_len: int = 10,
                 sync_window_ms: int = 50, 
                 logger: Optional[SyncedDataLogger] = None):
        """
        :param use_condition: 是否启用实时流
        :param action_history_len: 历史指令长度
        :param sync_window_ms: 帧同步时间窗口
        :param logger: 日志记录器
        """
        self._use_condition = use_condition
        self._sync_window_ms = sync_window_ms
        self._logger = logger
        self._latest_px4_time_est: int = 0
        self._time_lock = threading.Lock()
        self._last_update_wall_time: float = 0.0
        
        # --- Condition 控制 (实时控制) ---
        if self._use_condition:
            self._lock = threading.Lock()
            self._cv = threading.Condition(self._lock)
            self._has_new_data = False
            self._latest_obs: ObservationData = ObservationData()
            self._action_history: Deque[ActionData] = deque(maxlen=action_history_len)
            self._frame_cache = {
                'ATTITUDE': None,
                'LOCAL_POSITION_NED': None,
                'ACTUATOR_OUTPUT_STATUS': None
            }
    
    def on_new_observation(self, msg):
        """接收飞控消息"""
        msg_type = msg.get_type()
        current_px4_time = 0
        if hasattr(msg, 'time_boot_ms'):
            current_px4_time = msg.time_boot_ms
        elif hasattr(msg, 'time_usec'):
            current_px4_time = msg.time_usec // 1000
        
        if current_px4_time > 0:
            with self._time_lock:
                self._latest_px4_time_est = current_px4_time

        if self._use_condition:
            with self._lock:
                if msg_type in self._frame_cache:
                    self._frame_cache[msg_type] = msg
                self._try_emit_synced_frame()

    def on_new_action(self, action: ActionData):
        """发送了新的控制指令时调用此函数"""
        if action.time_boot_ms_est == 0:
            with self._time_lock:
                action.time_boot_ms_est = self._latest_px4_time_est

        if self._use_condition:
            with self._lock:
                self._action_history.append(action)

    # --- Condition 模式接口 ---
    def wait_for_synced_data(self, timeout: float = None) -> Optional[SyncedData]:
        """阻塞等待新数据，并自动对齐"""
        if not self._use_condition:
            return None

        with self._cv:
            signaled = self._cv.wait_for(lambda: self._has_new_data, timeout=timeout)
            if not signaled:
                return None
            
            self._has_new_data = False
            synced = SyncedData()
            synced.obs = ObservationData(**self._latest_obs.__dict__)
            if self._action_history:
                synced.last_action = ActionData(**self._action_history[-1].__dict__)
            return synced

    def get_latest_px4_time_ms(self) -> int:
        """获取当前最新的飞控时间戳"""
        with self._time_lock:
            return self._latest_px4_time_est

    def _try_emit_synced_frame(self):
        """消息集齐，推送完整信息"""
        cache = self._frame_cache
        if not (cache['ATTITUDE'] and cache['LOCAL_POSITION_NED'] and cache['ACTUATOR_OUTPUT_STATUS']):
            return
        
        t_att = cache['ATTITUDE'].time_boot_ms
        t_pos = cache['LOCAL_POSITION_NED'].time_boot_ms
        t_act = t_att
        if hasattr(cache['ACTUATOR_OUTPUT_STATUS'], 'time_boot_ms'):
            t_act = cache['ACTUATOR_OUTPUT_STATUS'].time_boot_ms
        
        times = [t_att, t_pos, t_act]
        if max(times) - min(times) > self._sync_window_ms:
            return 
        
        self._update_obs_with_msg(self._latest_obs, cache['ATTITUDE'], 'ATTITUDE')
        self._update_obs_with_msg(self._latest_obs, cache['LOCAL_POSITION_NED'], 'LOCAL_POSITION_NED')
        self._update_obs_with_msg(self._latest_obs, cache['ACTUATOR_OUTPUT_STATUS'], 'ACTUATOR_OUTPUT_STATUS')
        
        self._has_new_data = True
        self._last_update_wall_time = time.time()

        if self._logger:
            synced_data = SyncedData(
                obs=ObservationData(**self._latest_obs.__dict__),
                last_action=ActionData(**self._action_history[-1].__dict__) if self._action_history else None,
                received_wall_time=time.time()
            )
            self._logger.log_synced_data(synced_data)

        self._cv.notify()
        self._frame_cache = {k: None for k in self._frame_cache}

    def _update_obs_with_msg(self, obs: ObservationData, msg, msg_type: str):
        if msg_type == 'ATTITUDE':
            obs.time_boot_ms = msg.time_boot_ms
            obs.roll = msg.roll
            obs.pitch = msg.pitch
            obs.yaw = msg.yaw
            obs.rollspeed = msg.rollspeed
            obs.pitchspeed = msg.pitchspeed
            obs.yawspeed = msg.yawspeed
        elif msg_type == 'LOCAL_POSITION_NED':
            obs.time_boot_ms = msg.time_boot_ms
            obs.x = msg.x
            obs.y = msg.y
            obs.z = msg.z
            obs.vx = msg.vx
            obs.vy = msg.vy
            obs.vz = msg.vz
        elif msg_type == 'ACTUATOR_OUTPUT_STATUS':
            obs.motors = list(msg.actuator[:4])

# ==============================================
# 4. 仿真时间对齐
# ==============================================

class SimTimeKeeper:
    """仿真时间对齐工具类。替代 time.sleep() 和 asyncio.sleep()，让程序按照仿真时间流速运行。"""
    def __init__(self, data_sync: DroneDataSync):
        if not hasattr(data_sync, '_cv'):
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
        :param timeout: 现实世界中的超时时间（防止仿真卡死导致程序死等），None为无限等待
        :return: True 表示等待完成，False 表示现实超时
        """
        start_sim_ms = self._wait_for_first_tick()
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

    def _wait_for_first_tick(self) -> int:
        """等待直到接收到第一帧数据"""
        with self._lock:
            while self._data_sync.get_latest_px4_time_ms() == 0:
                self._cv.wait(0.1)
            return self._data_sync.get_latest_px4_time_ms()

    def _wait_until(self, target_ms: int, timeout: float = None) -> bool:
        """核心等待逻辑"""
        start_wall_time = time.time()
        with self._lock:
            while True:
                current_ms = self._data_sync.get_latest_px4_time_ms()
                
                if current_ms >= target_ms:
                    return True
                
                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_wall_time
                    if elapsed >= timeout:
                        return False
                    remaining_timeout = timeout - elapsed
                
                self._cv.wait_for(
                    lambda: self._data_sync.get_latest_px4_time_ms() >= target_ms,
                    timeout=remaining_timeout
                )

# ==============================================
# 5. 控制命令参数类 
# ==============================================

@dataclass
class ControlParams:
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.55