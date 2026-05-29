import time
import os
import asyncio
import logging
import threading
import queue
import math
from collections import deque
from typing import Optional, Deque, List

from pymavlink import mavutil

from .ctbr_tools import (
    ControlParams,
    DroneDataSync,
    ActionData,
    SimTimeKeeper,
    SyncedDataLogger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CTBRController")


class CTBRController:
    """
    CTBR MAVLink controller for PX4/Pegasus multi-vehicle simulation.
    """
    # MAVLink data messages used by DroneDataSync
    _SYNC_MSG_TYPES = {"ATTITUDE", "LOCAL_POSITION_NED", "ACTUATOR_OUTPUT_STATUS"}

    def __init__(
        self,
        connection_str='udp:0.0.0.0:14550',
        target_system=None,
        timeout=30,
        log_dir="./log_folder",
        log_subdir: str = None,
        enable_data_sync: bool = True,
        use_sync_condition: bool = True,
        enable_logging: bool = True,
        log_filename: str = None,
    ):
        self.master = mavutil.mavlink_connection(connection_str, timeout=timeout)

        # [MOD] 发送锁：send thread / offboard maintain / mode command 可能并发发送。
        self._send_lock = threading.Lock()

        # [MOD] ACK 与状态分发：接收线程唯一读取 socket，然后把命令 ACK 放入队列。
        self._ack_queue: "queue.Queue" = queue.Queue(maxsize=200)
        self._status_texts: Deque[str] = deque(maxlen=100)
        self._last_heartbeat = None
        self._last_recv_wall_time: float = 0.0
        self.data_sync = None
        self._armed: bool = False
        self._base_mode: int = 0
        self._custom_mode: int = 0
        self._last_takeoff_error: str = ""

        # [MOD-TAKEOFF] GLOBAL_POSITION_INT 缓存：
        # MAV_CMD_NAV_TAKEOFF 的 altitude 在 PX4 navigator 中不能简单当作本地相对高度。
        # 使用 COMMAND_INT + MAV_FRAME_GLOBAL_RELATIVE_ALT_INT 时需要当前经纬度。
        self._global_lat_e7: Optional[int] = None
        self._global_lon_e7: Optional[int] = None
        self._global_alt_m: Optional[float] = None          # AMSL, meters
        self._global_relative_alt_m: Optional[float] = None # above home, meters

        if target_system is not None:
            logger.info(f"Waiting for heartbeat from system {target_system}...")
            self._wait_heartbeat_from_system(target_system, timeout)
        else:
            logger.info("Waiting for heartbeat from any system...")
            self._wait_heartbeat_from_system(None, timeout)

        logger.info(
            f"Connected to PX4 (system {self.master.target_system}, "
            f"component {self.master.target_component})"
        )

        self._default_hover_x = 0.0
        self._default_hover_y = 0.0
        self._default_hover_z = -2.5

        self.is_offboard_running = False
        self.offboard_task = None
        self.is_monitor_ctbr_status = False
        self.is_monitor_ctbr_test_status = False

        # --- 数据监听相关状态 ---
        self.is_monitoring = False
        self.monitor_thread = None

        # --- 发送线程相关状态 ---
        self.is_sending = False
        self.send_thread = None
        self.send_frequency = 50
        self.current_params = ControlParams()
        self.param_lock = threading.Lock()

        # 初始化日志记录器
        self._logger: Optional[SyncedDataLogger] = None
        if enable_logging:
            if log_subdir is None:
                log_subdir = f"drone_{target_system}" if target_system is not None else "drone_unknown"

            vehicle_log_dir = os.path.join(log_dir, log_subdir)
            self._logger = SyncedDataLogger(log_dir=vehicle_log_dir, filename=log_filename)

        # 将 logger 传递给 DroneDataSync
        self.data_sync: Optional[DroneDataSync] = None
        if enable_data_sync:
            self.data_sync = DroneDataSync(
                use_condition=use_sync_condition,
                logger=self._logger,
            )

    # ----------------------------------------------------------------------
    # MAVLink receive / ACK / state helpers
    # ----------------------------------------------------------------------

    def _wait_heartbeat_from_system(self, target_system, timeout=30):
        """
        初始化阶段等待指定 system_id 的 HEARTBEAT。
        注意：此时监听线程尚未启动，因此这里可以直接 recv_match。
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            msg = self.master.recv_match(
                type='HEARTBEAT',
                blocking=True,
                timeout=1,
            )

            if msg is None:
                continue

            src_sys = msg.get_srcSystem()
            src_comp = msg.get_srcComponent()

            if target_system is None or src_sys == target_system:
                self.master.target_system = src_sys
                self.master.target_component = src_comp
                self._handle_heartbeat(msg)
                logger.info(
                    f"Connected to PX4 system {src_sys}, component {src_comp}"
                )
                return True

        raise TimeoutError(f"Timeout waiting for heartbeat from system {target_system}")

    def _handle_heartbeat(self, msg):
        """[MOD] 统一维护 armed/base_mode/custom_mode 状态。"""
        self._last_heartbeat = msg
        self._last_recv_wall_time = time.time()
        self._base_mode = int(getattr(msg, "base_mode", 0))
        self._custom_mode = int(getattr(msg, "custom_mode", 0))
        self._armed = bool(
            self._base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        self._push_runtime_status_to_logger()

    def _handle_status_text(self, msg):
        """[MOD] 保存并打印 PX4 STATUSTEXT，方便定位 Battery unhealthy / mode reject 等问题。"""
        text = getattr(msg, "text", "")
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        text = str(text).replace("\x00", "").strip()
        if not text:
            return

        self._status_texts.append(text)

        severity = int(getattr(msg, "severity", 6))
        if severity <= mavutil.mavlink.MAV_SEVERITY_WARNING:
            logger.warning(f"[PX4 STATUSTEXT] {text}")
        else:
            logger.info(f"[PX4 STATUSTEXT] {text}")
        self._push_runtime_status_to_logger()

    def _handle_global_position_int(self, msg):
        """[MOD-TAKEOFF] 缓存全局位置，用于构造不会被 PX4 误判高度的起飞命令。"""
        try:
            self._global_lat_e7 = int(msg.lat)
            self._global_lon_e7 = int(msg.lon)
            self._global_alt_m = float(msg.alt) / 1000.0
            self._global_relative_alt_m = float(msg.relative_alt) / 1000.0

            logger.debug(
                f"[GLOBAL] "
                f"lat={self._global_lat_e7}, "
                f"lon={self._global_lon_e7}, "
                f"amsl={self._global_alt_m:.2f}, "
                f"rel={self._global_relative_alt_m:.2f}")

        except Exception as e:
            logger.debug(f"解析 GLOBAL_POSITION_INT 失败: {e}")

    def _handle_command_ack(self, msg):
        """[MOD] COMMAND_ACK 进入队列，由命令函数按 command id 等待。"""
        try:
            self._ack_queue.put_nowait(msg)
        except queue.Full:
            # 队列满时丢弃最旧 ACK，保留较新的 ACK。
            try:
                self._ack_queue.get_nowait()
                self._ack_queue.put_nowait(msg)
            except queue.Empty:
                pass

    def _handle_incoming_message(self, msg):
        """
        [MOD] 唯一 MAVLink 消息分发入口。
        所有从 socket 读到的消息都从这里进入状态机/同步器。
        """
        if msg is None:
            return

        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            return

        if hasattr(msg, "get_srcSystem") and msg.get_srcSystem() != self.master.target_system:
            return

        if msg_type == "HEARTBEAT":
            self._handle_heartbeat(msg)
            return

        if msg_type == "COMMAND_ACK":
            self._handle_command_ack(msg)
            return

        if msg_type == "STATUSTEXT":
            self._handle_status_text(msg)
            return

        if msg_type == "GLOBAL_POSITION_INT":
            self._handle_global_position_int(msg)
            logger.debug(
                f"[GLOBAL] lat={self._global_lat_e7}, lon={self._global_lon_e7}, "
                f"alt_amsl={self._global_alt_m}, rel_alt={self._global_relative_alt_m}"
            )
            return

        if msg_type == "ATTITUDE":
            logger.debug(
                f"[PX4时间: {msg.time_boot_ms:>8}ms] [角速率] "
                f"Roll: {msg.rollspeed:+.4f} | "
                f"Pitch: {msg.pitchspeed:+.4f} | "
                f"Yaw: {msg.yawspeed:+.4f} rad/s"
            )

        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            px4_time = getattr(msg, "time_usec", 0) // 1000
            motors = msg.actuator[:4]
            logger.debug(
                f"[PX4时间: {px4_time:>8}ms] [执行器] "
                f"电机: {[f'{m:.1f}' for m in motors]}"
            )

        elif msg_type == "LOCAL_POSITION_NED":
            relative_alt = -msg.z
            logger.debug(
                f"[PX4时间: {msg.time_boot_ms:>8}ms] [📌 坐标] "
                f"X(前): {msg.x:.2f} | Y(右): {msg.y:.2f} | "
                f"高度: {relative_alt:.2f} m"
            )

        # [MOD] 只把同步器关心的数据消息送入 DroneDataSync。
        if self.data_sync and msg_type in self._SYNC_MSG_TYPES:
            self.data_sync.on_new_observation(msg)

    def _clear_command_ack_queue(self, command: Optional[int] = None):
        """
        [MOD] 清空 ACK 队列。
        command=None 时清空全部；指定 command 时清空该命令旧 ACK，其他 ACK 丢弃。
        对当前控制器来说，旧 ACK 均不应参与新命令判断。
        """
        while True:
            try:
                ack = self._ack_queue.get_nowait()
            except queue.Empty:
                break

    def _wait_command_ack(self, command: int, timeout: float = 3.0):
        """
        [MOD] 等待指定 MAV_CMD 的 COMMAND_ACK。
        注意：只有 _recv_data_loop 读取 socket，本函数只读内部 ACK 队列。
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                ack = self._ack_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue

            if int(getattr(ack, "command", -1)) == int(command):
                return ack

        return None

    def _result_name(self, result: int) -> str:
        """[MOD] 将 MAV_RESULT 数值转为可读名称。"""
        for name in dir(mavutil.mavlink):
            if name.startswith("MAV_RESULT_") and getattr(mavutil.mavlink, name) == result:
                return name
        return str(result)

    def _recent_status_text(self, n: int = 5) -> List[str]:
        return list(self._status_texts)[-n:]

    def _wait_until_armed_state(self, desired_armed: bool, timeout: float = 5.0) -> bool:
        """[MOD] 通过接收线程维护的 HEARTBEAT armed 状态确认解锁/上锁。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._armed == desired_armed:
                return True
            time.sleep(0.05)
        return False

    # ----------------------------------------------------------------------
    # High-level flight control
    # ----------------------------------------------------------------------

    def arm_drone(self, timeout: int = 5, use_sim_time: bool = True) -> bool:
        """
        解锁电机。

        [MOD] 不再在这里 recv_match()，避免和监听线程抢 MAVLink 消息。
        判断逻辑改为：
        1. 发送 MAV_CMD_COMPONENT_ARM_DISARM
        2. 等待 COMMAND_ACK
        3. 等待 HEARTBEAT 中的 armed bit 变为 True
        """
        logger.info("正在发送电机解锁命令...")

        if not self.data_sync or not self.is_monitoring:
            logger.error("解锁失败：请先启动数据监听")
            return False

        if self._armed:
            logger.info("✅ 电机已经处于解锁状态")
            return True

        self._clear_command_ack_queue()

        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,  # 解锁
                0, 0, 0, 0, 0, 0,
            )

        ack = self._wait_command_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            timeout=timeout,
        )

        if ack is None:
            logger.error("❌ 电机解锁失败：未收到 COMMAND_ACK")
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
            return False

        if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            logger.error(
                f"❌ 电机解锁被 PX4 拒绝："
                f"{self._result_name(ack.result)} ({ack.result})"
            )
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
            return False

        if self._wait_until_armed_state(True, timeout=timeout):
            logger.info("✅ 电机解锁成功！")
            return True

        logger.error("❌ 解锁 ACK 已接受，但 HEARTBEAT 未显示 armed")
        logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
        return False

    def disarm_drone(self, timeout: int = 5) -> bool:
        """[MOD] 可选工具：上锁电机，便于 episode reset/cleanup。"""
        logger.info("正在发送电机上锁命令...")

        self._clear_command_ack_queue()

        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,  # 上锁
                0, 0, 0, 0, 0, 0,
            )

        ack = self._wait_command_ack(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            timeout=timeout,
        )

        if ack is None:
            logger.error("上锁失败：未收到 COMMAND_ACK")
            return False

        if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            logger.error(
                f"上锁被 PX4 拒绝：{self._result_name(ack.result)} ({ack.result})"
            )
            return False

        if self._wait_until_armed_state(False, timeout=timeout):
            logger.info("✅ 电机上锁成功")
            return True

        logger.warning("上锁 ACK 已接受，但 HEARTBEAT 未确认 disarmed")
        return False

    def _wait_for_global_position(self, timeout: float = 5.0) -> bool:
        """[MOD-TAKEOFF] 等待 GLOBAL_POSITION_INT，起飞命令需要用它构造相对高度帧。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (
                self._global_lat_e7 is not None
                and self._global_lon_e7 is not None
                and self._global_alt_m is not None
            ):
                return True
            time.sleep(0.05)
        return False

    def _send_takeoff_command(self, target_altitude: float, timeout: float = 5.0):
        """
        发送 PX4 起飞命令并等待 ACK。

        target_altitude: 相对 Home 的目标起飞高度，单位 m。
        实际发送给 PX4 的 COMMAND_LONG param7 使用 AMSL 绝对高度，
        避免 PX4 main/1.17 中 COMMAND_INT + GLOBAL_RELATIVE_ALT_INT
        被接受但不实际爬升的问题。
        """
        self._last_takeoff_error = ""

        accepted_results = {
            mavutil.mavlink.MAV_RESULT_ACCEPTED,
            getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5),
        }

        if target_altitude <= 0:
            self._last_takeoff_error = f"非法起飞高度: {target_altitude}"
            logger.error(self._last_takeoff_error)
            return False, None

        has_global = self._wait_for_global_position(timeout=8.0)
        if not has_global:
            self._last_takeoff_error = (
                "未收到 GLOBAL_POSITION_INT，无法计算 AMSL 起飞高度。"
                "请确认 start_monitoring() 包含 msg_id=33。"
            )
            logger.error(self._last_takeoff_error)
            return False, None

        if self._global_alt_m is None:
            self._last_takeoff_error = "GLOBAL_POSITION_INT 中 AMSL 高度无效"
            logger.error(self._last_takeoff_error)
            return False, None

        current_amsl = float(self._global_alt_m)

        # 更稳妥：先估算 home_amsl，再加目标相对高度
        if self._global_relative_alt_m is not None:
            home_amsl = current_amsl - float(self._global_relative_alt_m)
        else:
            home_amsl = current_amsl

        target_amsl = home_amsl + float(target_altitude)

        lat_deg = (
            float(self._global_lat_e7) / 1e7
            if self._global_lat_e7 is not None
            else 0.0
        )
        lon_deg = (
            float(self._global_lon_e7) / 1e7
            if self._global_lon_e7 is not None
            else 0.0
        )

        logger.info(
            "[TAKEOFF DEBUG] "
            f"current_amsl={current_amsl:.2f}m, "
            f"current_rel={self._global_relative_alt_m}, "
            f"home_amsl={home_amsl:.2f}m, "
            f"target_rel={target_altitude:.2f}m, "
            f"target_amsl={target_amsl:.2f}m"
        )

        logger.info(
            "发送自动起飞指令 MAV_CMD_NAV_TAKEOFF "
            f"(COMMAND_LONG, target_amsl={target_amsl:.2f}m)"
        )

        self._clear_command_ack_queue()

        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0.0,           # param1: pitch，MC 通常忽略
                0.0,
                0.0,
                float("nan"),  # param4: yaw，NaN=保持当前航向
                lat_deg,
                lon_deg,
                target_amsl,   # param7: AMSL 绝对高度
            )

        ack = self._wait_command_ack(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            timeout=timeout,
        )

        if ack is None:
            self._last_takeoff_error = "未收到 TAKEOFF COMMAND_ACK"
            logger.error(self._last_takeoff_error)
            return False, None

        if ack.result not in accepted_results:
            self._last_takeoff_error = (
                f"TAKEOFF 被 PX4 拒绝: {self._result_name(ack.result)} ({ack.result})"
            )
            logger.error(self._last_takeoff_error)
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
            return False, ack

        logger.info(
            f"TAKEOFF 命令已被 PX4 接受: {self._result_name(ack.result)} ({ack.result})"
        )

        return True, ack

    def auto_takeoff(self, target_altitude: float = 6.0, timeout: int = 15, use_sim_time: bool = True) -> bool:
        """
        自动起飞。
        """
        logger.info(f"开始起飞，目标高度: {target_altitude}m")

        if not hasattr(self, "data_sync") or self.data_sync is None:
            logger.error("起飞失败：数据同步模块未初始化！请先调用 start_monitoring()")
            return False

        if not self.is_monitoring:
            logger.error("起飞失败：数据监听未启动！请先调用 start_monitoring()")
            return False

        # 等待收到有效飞控时间/状态数据
        start_wait = time.time()
        while time.time() - start_wait < 5.0:
            if self.data_sync.get_latest_px4_time_ms() > 0:
                break
            time.sleep(0.05)
        else:
            logger.error("起飞失败：未收到任何飞控数据！请检查飞控连接/消息流")
            return False

        time_keeper = None
        if use_sim_time and self.data_sync._use_condition:
            try:
                time_keeper = self.get_sim_time_keeper()
                logger.info("已启用仿真时间对齐模式")
            except RuntimeError:
                logger.warning("SimTimeKeeper 不可用，将回退到系统时间")
                time_keeper = None

        if not self.arm_drone(timeout=5, use_sim_time=use_sim_time):
            logger.error("起飞失败：电机解锁失败")
            return False

        takeoff_ok, ack = self._send_takeoff_command(
            target_altitude=target_altitude,
            timeout=5.0,
        )

        if ack is None:
            logger.error(f"起飞失败：{self._last_takeoff_error or '未收到 TAKEOFF COMMAND_ACK'}")
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
            return False

        if not takeoff_ok:
            logger.error(
                f"起飞命令被 PX4 拒绝：{self._result_name(ack.result)} ({ack.result})"
            )
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
            return False

        if time_keeper:
            start_sim_ms = time_keeper.now_ms()
            timeout_ms = int(timeout * 1000)

            def not_timeout():
                return time_keeper.now_ms() - start_sim_ms < timeout_ms
        else:
            start_time = time.time()

            def not_timeout():
                return time.time() - start_time < timeout
        
        while not_timeout():
            with self.data_sync._lock:
                current_alt = -self.data_sync._latest_obs.z

            logger.info(
                f"起飞中... 当前高度: {current_alt:.2f}m / "
                f"目标: {target_altitude:.2f}m"
            )

            if current_alt >= target_altitude * 0.9:
                logger.info(f"起飞完成，已到达高度: {current_alt:.2f}m")
                return True

            if time_keeper:
                time_keeper.wait(0.2, timeout=2.0)
            else:
                time.sleep(0.2)

        logger.error(f"起飞超时，{timeout}秒内未到达目标高度")
        logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")
        return False

    # ----------------------------------------------------------------------
    # Logging / time helpers
    # ----------------------------------------------------------------------

    def start_logging(self):
        """启动数据日志记录"""
        if self._logger:
            self._logger.start()

    def stop_logging(self):
        """停止数据日志记录，会强制刷新剩余数据"""
        if self._logger:
            self._logger.stop()

    def get_sim_time_keeper(self) -> SimTimeKeeper:
        """获取仿真时间管理器实例"""
        if not self.data_sync:
            raise RuntimeError("必须开启 enable_data_sync=True 才能使用 SimTimeKeeper")
        return SimTimeKeeper(self.data_sync)

    # ----------------------------------------------------------------------
    # OFFBOARD / setpoint control
    # ----------------------------------------------------------------------

    def send_hover_setpoint(self, x=0, y=0, z=-10):
        """发送位置 hold setpoint，用于 OFFBOARD 预热/保活。"""
        with self._send_lock:
            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                8184,
                x, y, z,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
            )

    def change_control_mode(
        self,
        mode=6,
        is_maintain_offboard=False,
        default_x=0.0,
        default_y=0.0,
        default_z=-2.5,
        wait_for_data_timeout=3.0,
    ):
        """
        改变控制模式。

        [MOD] 切 OFFBOARD 前仍然发送 40 帧 setpoint 预热；
              模式切换 ACK 通过内部 ACK 队列等待，不再直接 recv_match。
        """
        initial_x, initial_y, initial_z = default_x, default_y, default_z
        use_default = True

        if self.data_sync and self.data_sync._use_condition:
            logger.info(f"正在等待飞控数据 (超时: {wait_for_data_timeout}s)...")
            start_wait_time = time.time()
            found_valid_data = False

            while (time.time() - start_wait_time) < wait_for_data_timeout:
                is_fresh = False

                with self.data_sync._lock:
                    last_update_time = getattr(self.data_sync, "_last_update_wall_time", 0.0)
                    has_data = self.data_sync._latest_obs.time_boot_ms > 0
                    is_new_data = last_update_time > start_wait_time

                    if has_data and is_new_data:
                        initial_x = self.data_sync._latest_obs.x
                        initial_y = self.data_sync._latest_obs.y
                        initial_z = self.data_sync._latest_obs.z
                        is_fresh = True

                if is_fresh:
                    use_default = False
                    found_valid_data = True
                    logger.info(
                        f"成功获取当前位置: "
                        f"X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}"
                    )
                    break

                time.sleep(0.02)

            if not found_valid_data:
                logger.warning(
                    f"等待数据超时，将使用默认/输入位置: "
                    f"X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}"
                )

        if use_default:
            logger.info(
                f"保活指令，使用默认输入位置: "
                f"X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}"
            )

        # mode=6 通常为 PX4 OFFBOARD custom mode。
        if mode == 6:
            for _ in range(40):
                self.send_hover_setpoint(initial_x, initial_y, initial_z)
                time.sleep(0.05)

        self._clear_command_ack_queue()

        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode,
                0, 0, 0, 0, 0, 0,
            )

        ack = self._wait_command_ack(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            timeout=3.0,
        )

        success = False
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            logger.info(f"成功转换至模式 {mode}")
            success = True
        else:
            if ack is None:
                logger.error(f"转换至模式 {mode} 失败：未收到 COMMAND_ACK")
            else:
                logger.error(
                    f"转换至模式 {mode} 失败："
                    f"{self._result_name(ack.result)} ({ack.result})"
                )
            logger.error(f"最近 PX4 STATUSTEXT: {self._recent_status_text()}")

        if success and is_maintain_offboard and mode == 6:
            self.start_offboard_maintain(default_x, default_y, default_z)

        return success

    async def _offboard_maintain_coroutine(self):
        """OFFBOARD 异步保活协程。"""
        try:
            while self.is_offboard_running:
                target_x = self._default_hover_x
                target_y = self._default_hover_y
                target_z = self._default_hover_z

                if self.data_sync and self.data_sync._use_condition:
                    with self.data_sync._lock:
                        if self.data_sync._latest_obs.time_boot_ms > 0:
                            target_x = self.data_sync._latest_obs.x
                            target_y = self.data_sync._latest_obs.y
                            target_z = self.data_sync._latest_obs.z

                logger.debug(
                    f"保活发送位置指令: "
                    f"X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}"
                )
                self.send_hover_setpoint(target_x, target_y, target_z)
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.info("OFFBOARD 保活任务已关闭")
        finally:
            logger.info("OFFBOARD 保活协程已退出")

    def start_offboard_maintain(self, default_x=None, default_y=None, default_z=None):
        """启动 OFFBOARD 保活。"""
        if not self.is_offboard_running:
            if default_x is not None:
                self._default_hover_x = default_x
            if default_y is not None:
                self._default_hover_y = default_y
            if default_z is not None:
                self._default_hover_z = default_z

            self.is_offboard_running = True
            self.offboard_task = asyncio.create_task(self._offboard_maintain_coroutine())
            logger.info("OFFBOARD 保活任务已启动")

    def stop_offboard_maintain(self):
        """停止 OFFBOARD 保活。"""
        self.is_offboard_running = False
        if self.offboard_task:
            self.offboard_task.cancel()
            logger.info("OFFBOARD 保活任务已关闭")

    def set_ctbr_parameters_send(
        self,
        body_roll_rate=0.0,
        body_pitch_rate=0.0,
        body_yaw_rate=0.0,
        thrust=0.0,
    ):
        """
        发送 body-rate + thrust CTBR 控制指令。

        [MOD] 原代码 type_mask=16 且 q=[0,0,0,0] 不合法。
              这里使用 ATTITUDE_IGNORE，只控制 body rates + thrust；
              q 使用单位四元数 [1,0,0,0]。
        """
        if self.is_offboard_running:
            logger.info("OFFBOARD 保活任务已自动关闭...")
            self.stop_offboard_maintain()

        local_time = time.time()

        type_mask = getattr(
            mavutil.mavlink,
            "ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE",
            128,
        )

        with self._send_lock:
            self.master.mav.set_attitude_target_send(
                0,
                self.master.target_system,
                self.master.target_component,
                type_mask,
                [1.0, 0.0, 0.0, 0.0],
                body_roll_rate,
                body_pitch_rate,
                body_yaw_rate,
                thrust,
            )

        if self.data_sync:
            action = ActionData(
                time_sent_local=local_time,
                body_roll_rate=body_roll_rate,
                body_pitch_rate=body_pitch_rate,
                body_yaw_rate=body_yaw_rate,
                thrust=thrust,
            )
            self.data_sync.on_new_action(action)

    def set_ctbr_parameters_continuously(
        self,
        body_roll_rate=[0.0, 0.0],
        body_pitch_rate=[0.0, 0.0],
        body_yaw_rate=[0.0, 0.0],
        thrust=[0.0, 0.0],
        frequency=20,
    ):
        """发送 CTBR 控制指令（连续序列）。"""
        for roll, pitch, yaw, t in zip(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust):
            self.set_ctbr_parameters_send(roll, pitch, yaw, t)
            time.sleep(1.0 / frequency)

    def set_ctbr_parameters_repeatly(
        self,
        body_roll_rate=0.0,
        body_pitch_rate=0.0,
        body_yaw_rate=0.0,
        thrust=0.0,
        frequency=20,
        repeat_times=10,
    ):
        """发送 CTBR 控制指令（重复）。"""
        for _ in range(repeat_times):
            self.set_ctbr_parameters_send(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust)
            time.sleep(1.0 / frequency)

    # ----------------------------------------------------------------------
    # Message stream / receive thread
    # ----------------------------------------------------------------------

    def request_message_stream(self, msg_id, freq_hz=10):
        """请求飞控以指定频率发送特定消息。"""
        interval_us = int(1e6 / freq_hz) if freq_hz > 0 else -1

        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0, 0,
            )

    def _recv_data_loop(self):
        """
        [MOD] 唯一 MAVLink 接收线程。
        后续不要在 arm/takeoff/change_mode 等函数里直接 recv_match。
        """
        while self.is_monitoring:
            msg = self.master.recv_match(blocking=False)

            if not msg:
                time.sleep(0.001)
                continue

            self._handle_incoming_message(msg)

    def start_monitoring(self, message_ids=None, freq_hz=20):
        """启动数据监听线程。"""
        if message_ids is None:
            message_ids = [30, 375, 32, 33]  # ATTITUDE, ACTUATOR_OUTPUT_STATUS, LOCAL_POSITION_NED, GLOBAL_POSITION_INT

        if not self.is_monitoring:
            for msg_id in message_ids:
                self.request_message_stream(msg_id, freq_hz)
            logger.info("已请求数据流")

            logger.info("正在清空 MAVLink 旧数据缓冲区...")
            start_flush_time = time.time()
            flushed_packets = 0
            last_msg_time = time.time()

            while True:
                msg = self.master.recv_match(blocking=False)
                if msg:
                    flushed_packets += 1
                    last_msg_time = time.time()
                    # [MOD] flush 期间仍处理 HEARTBEAT/STATUSTEXT，避免丢失 armed 状态和错误原因。
                    if msg.get_type() in {"HEARTBEAT", "STATUSTEXT", "GLOBAL_POSITION_INT"}:
                        self._handle_incoming_message(msg)
                else:
                    if time.time() - last_msg_time > 0.2:
                        break
                    time.sleep(0.01)

                if time.time() - start_flush_time > 1.0:
                    break

            logger.info(f"缓冲区清空完成，丢弃了 {flushed_packets} 条旧消息")

            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._recv_data_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """停止数据监听线程。"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            logger.info("数据监听已停止")

    # ----------------------------------------------------------------------
    # CTBR send thread
    # ----------------------------------------------------------------------

    def _send_ctbr_control_loop(self):
        """线程内部函数：循环读取 current_params 并发送。"""
        logger.info(f"发送线程启动，频率: {self.send_frequency}Hz")
        while self.is_sending:
            with self.param_lock:
                roll = self.current_params.body_roll_rate
                pitch = self.current_params.body_pitch_rate
                yaw = self.current_params.body_yaw_rate
                thrust = self.current_params.thrust

            self.set_ctbr_parameters_send(roll, pitch, yaw, thrust)
            time.sleep(1.0 / self.send_frequency)

        logger.info("发送线程已停止")

    def start_ctbr_send_thread(self, frequency=50):
        """启动 CTBR 发送线程。"""
        if not self.is_sending:
            self.send_frequency = frequency
            self.is_sending = True
            self.send_thread = threading.Thread(target=self._send_ctbr_control_loop, daemon=True)
            self.send_thread.start()

    def stop_ctbr_send_thread(self):
        """停止 CTBR 发送线程。"""
        if self.is_sending:
            self.is_sending = False
            if self.send_thread:
                self.send_thread.join(timeout=2.0)

    def update_ctbr_send_params(
        self,
        body_roll_rate=None,
        body_pitch_rate=None,
        body_yaw_rate=None,
        thrust=None,
    ):
        """更新 CTBR 发送参数。"""
        with self.param_lock:
            if body_roll_rate is not None:
                self.current_params.body_roll_rate = body_roll_rate
            if body_pitch_rate is not None:
                self.current_params.body_pitch_rate = body_pitch_rate
            if body_yaw_rate is not None:
                self.current_params.body_yaw_rate = body_yaw_rate
            if thrust is not None:
                self.current_params.thrust = thrust

    def check_crash_or_failure(
        self,
        min_altitude: float = 0.20,
        max_tilt_deg: float = 75.0,
        max_body_rate: float = 8.0,
        max_down_speed: float = 5.0,
    ) -> tuple[bool, str]:
        """
        返回: (是否异常终止, 原因)
        """
        if not self.data_sync:
            return True, "no_data_sync"

        obs = self.data_sync.get_latest_observation()

        alt = -obs.z
        tilt = math.degrees(math.sqrt(obs.roll ** 2 + obs.pitch ** 2))
        max_rate = max(abs(obs.rollspeed), abs(obs.pitchspeed), abs(obs.yawspeed))

        if not self._armed:
            return True, "disarmed"

        recent_text = " | ".join(self._recent_status_text(10)).lower()
        if "failsafe" in recent_text or "rtl" in recent_text or "land" in recent_text:
            return True, f"px4_status: {recent_text}"

        if alt < min_altitude and abs(obs.vz) > 1.0:
            return True, f"near_ground_alt={alt:.2f}"

        if tilt > max_tilt_deg:
            return True, f"tilt_too_large={tilt:.1f}deg"

        if max_rate > max_body_rate:
            return True, f"body_rate_too_large={max_rate:.2f}rad/s"

        # NED 中 vz > 0 表示向下
        if obs.vz > max_down_speed:
            return True, f"falling_fast_vz={obs.vz:.2f}"

        return False, "ok"

    def recover_to_local_position(
        self,
        x: float,
        y: float,
        z: float,
        timeout: float = 8.0,
        tolerance: float = 0.5,
    ) -> bool:
        """
        用位置 setpoint 把飞机拉回指定 local NED 坐标。
        注意：z 是 NED，高度 10m 对应 z=-10。
        """
        if self.is_sending:
            self.stop_ctbr_send_thread()

        # 继续保持 OFFBOARD 位置控制
        self.change_control_mode(
            mode=6,
            is_maintain_offboard=True,
            default_x=x,
            default_y=y,
            default_z=z,
        )

        start = time.time()

        while time.time() - start < timeout:
            self.send_hover_setpoint(x, y, z)

            obs = self.data_sync.get_latest_observation()
            err = math.sqrt(
                (obs.x - x) ** 2 +
                (obs.y - y) ** 2 +
                (obs.z - z) ** 2
            )

            if err < tolerance:
                logger.info(f"已恢复到悬停点，误差 {err:.2f}m")
                return True

            time.sleep(0.05)

        logger.warning("恢复悬停点超时")
        return False

    def _flight_mode_name(self) -> str:
        if self._custom_mode == 6:
            return "OFFBOARD"
        if self._custom_mode == 4:
            return "AUTO"
        if self._custom_mode == 3:
            return "POSCTL"
        if self._custom_mode == 2:
            return "ALTCTL"
        if self._custom_mode == 1:
            return "MANUAL"
        return str(self._custom_mode)


    def _push_runtime_status_to_logger(self):
        if not hasattr(self, "data_sync"):
            return

        if self.data_sync is None:
            return

        recent_text = " | ".join(self._recent_status_text(10))
        recent_lower = recent_text.lower()

        failsafe = (
            "failsafe" in recent_lower
            or "rtl" in recent_lower
            or "battery warning" in recent_lower
            or "land" in recent_lower
        )

        self.data_sync.update_runtime_status(
            armed=self._armed,
            base_mode=self._base_mode,
            custom_mode=self._custom_mode,
            flight_mode=self._flight_mode_name(),
            failsafe=failsafe,
            last_status_text=recent_text,
        )


    def set_episode(self, episode_id: int, phase: str = "collect", step_id: int = 0):
        if self.data_sync:
            self.data_sync.set_episode(episode_id, phase=phase, step_id=step_id)


    def set_episode_step(self, step_id: int):
        if self.data_sync:
            self.data_sync.set_step(step_id)


    def set_episode_phase(self, phase: str):
        if self.data_sync:
            self.data_sync.set_phase(phase)


    def mark_episode_done(self, reason: str = "normal", crashed: bool = False):
        if self.data_sync:
            self.data_sync.mark_done(done_reason=reason, crashed=crashed)


    def clear_episode_done(self):
        if self.data_sync:
            self.data_sync.clear_done()

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------

    def cleanup(self):
        """清理与关闭所有功能。"""
        logger.info("🔧 开始执行 CTBRController 资源清理...")

        if self.is_sending:
            logger.info("🛑 正在停止控制指令发送线程...")
            self.stop_ctbr_send_thread()

        if self.is_offboard_running:
            logger.info("🛑 正在停止 OFFBOARD 保活协程...")
            self.stop_offboard_maintain()

        if self.is_monitoring:
            logger.info("🛑 正在停止数据监听线程...")
            self.stop_monitoring()

        if self._logger:
            logger.info("🛑 正在停止日志记录器并刷新缓冲区...")
            self._logger.stop()

        if hasattr(self, "master") and self.master:
            logger.info("🔌 正在关闭 MAVLink 连接...")
            try:
                self.master.close()
            except Exception as e:
                logger.warning(f"关闭 MAVLink 连接时出现警告: {e}")

        logger.info("✅ CTBRController 资源清理完成")
