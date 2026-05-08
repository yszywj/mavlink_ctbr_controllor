from pymavlink import mavutil
import time
import asyncio
import os
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import threading
from .ctbr_tools import ControlParams, DroneDataSync, ActionData, ObservationData, SimTimeKeeper, SyncedDataLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CTBRController")

class CTBRController:
    def __init__(self, connection_str='udp:0.0.0.0:14550', timeout=30, log_dir="./log_folder", thrust_output_index=None, 
                enable_data_sync: bool = True, 
                use_sync_condition: bool = True,
                enable_logging: bool = True):
        self.master = mavutil.mavlink_connection(connection_str, timeout=timeout)
        logger.info("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        logger.info(f"Connected to PX4 (system {self.master.target_system}, component {self.master.target_component})")

        self._default_hover_x = 0.0
        self._default_hover_y = 0.0
        self._default_hover_z = -2.5

        self.is_offboard_running = False
        self.offboard_task = None
        self.is_monitor_ctbr_status = False
        self.is_monitor_ctbr_test_status = False

        self.thrust_output_index = thrust_output_index

        # --- 数据监听相关状态 ---
        self.is_monitoring = False
        self.monitor_thread = None

        # --- 发送线程相关状态 ---
        self.is_sending = False
        self.send_thread = None
        self.send_frequency = 50  # 默认50Hz
        # 实例化参数对象，并加一把锁保护
        self.current_params = ControlParams()
        self.param_lock = threading.Lock()

        # 初始化日志记录器
        self._logger: Optional[SyncedDataLogger] = None
        if enable_logging:
            self._logger = SyncedDataLogger(log_dir=log_dir)

        # 将 logger 传递给 DroneDataSync
        self.data_sync: Optional[DroneDataSync] = None
        if enable_data_sync:
            self.data_sync = DroneDataSync(
                use_condition=use_sync_condition, 
                logger=self._logger
            )

    # 便捷控制日志启停的方法
    def start_logging(self):
        """启动数据日志记录"""
        if self._logger:
            self._logger.start()

    def stop_logging(self):
        """停止数据日志记录 (会强制刷新缓冲区)"""
        if self._logger:
            self._logger.stop()

    # 新增：便捷获取时间守护者的方法
    def get_sim_time_keeper(self) -> SimTimeKeeper:
        """获取仿真时间管理器实例"""
        if not self.data_sync:
            raise RuntimeError("必须开启 enable_data_sync=True 才能使用 SimTimeKeeper")
        return SimTimeKeeper(self.data_sync)

    # OFFBOARD 保活指令
    def send_hover_setpoint(self, x=0, y=0, z=-10):
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            8184,
            x, y, z,                               # 目标位置坐标
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0
        )
        
    # 改变控制模式
    def change_control_mode(self, mode=6, is_maintain_offboard=False, default_x=0.0, default_y=0.0, default_z=-2.5, wait_for_data_timeout=1.0):
        initial_x, initial_y, initial_z = default_x, default_y, default_z
        use_default = True
        
        if self.data_sync and self.data_sync._use_condition:
            logger.info(f"⏳ 正在等待飞控数据 (超时: {wait_for_data_timeout}s)...")
            start_wait_time = time.time()
            found_valid_data = False
            
            while (time.time() - start_wait_time) < wait_for_data_timeout:
                temp_x, temp_y, temp_z = initial_x, initial_y, initial_z
                is_fresh = False
                
                with self.data_sync._lock:
                    # 安全获取属性，防止 AttributeError
                    last_update_time = getattr(self.data_sync, '_last_update_wall_time', 0.0)
                    
                    # 条件A: 有数据
                    has_data = self.data_sync._latest_obs.time_boot_ms > 0
                    
                    # 条件B: 数据更新时间 晚于 我们开始等待的时间
                    # (这确保了数据是在我们开始这个函数之后才收到的新数据)
                    is_new_data = last_update_time > start_wait_time
                    
                    if has_data and is_new_data:
                        temp_x = self.data_sync._latest_obs.x
                        temp_y = self.data_sync._latest_obs.y
                        temp_z = self.data_sync._latest_obs.z
                        is_fresh = True
                
                if is_fresh:
                    initial_x, initial_y, initial_z = temp_x, temp_y, temp_z
                    use_default = False
                    found_valid_data = True
                    logger.info(f"✅ 成功获取当前位置: X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}")
                    break
                
                time.sleep(0.02)
            
            if not found_valid_data:
                logger.warning(f"⏱️ 等待数据超时，将使用默认/输入位置: X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}")
        # ==============================================
        # 新增结束
        # ==============================================
        if use_default:
            logger.info(f"🚀保活指令，使用默认输入位置: X={initial_x:.2f}, Y={initial_y:.2f}, Z={initial_z:.2f}")

        if mode == 6:
            for _ in range(40):
                self.send_hover_setpoint(initial_x, initial_y, initial_z)
                time.sleep(0.05)

        self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode, 0,0,0,0,0,0
            )
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
        if ack and ack.result == 0:
            logger.info(f"Switched to mode {mode} successfully.")
        else:
            logger.error(f"Failed to switch to mode {mode}.") 
        
        if is_maintain_offboard and mode == 6:
            self.start_offboard_maintain(default_x, default_y, default_z)

    # OFFBOARD 异步保活协程
    async def _offboard_maintain_coroutine(self):
        try:
            while self.is_offboard_running:
                # 1. 确定目标坐标
                target_x = self._default_hover_x
                target_y = self._default_hover_y
                target_z = self._default_hover_z

                # 2. 尝试从 data_sync 获取最新位置 (线程安全读取)
                if self.data_sync and self.data_sync._use_condition:
                    with self.data_sync._lock:
                        # 检查时间戳确保数据是有效的
                        if self.data_sync._latest_obs.time_boot_ms > 0:
                            target_x = self.data_sync._latest_obs.x
                            target_y = self.data_sync._latest_obs.y
                            target_z = self.data_sync._latest_obs.z

                logger.debug(f"保活发送位置指令: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")

                # 3. 发送指令
                self.send_hover_setpoint(target_x, target_y, target_z)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logger.info("OFFBOARD keep-alive task cancelled gracefully")
        finally:
            logger.info("OFFBOARD maintain coroutine exited")

    # 启动 OFFBOARD 保活
    def start_offboard_maintain(self, default_x=None, default_y=None, default_z=None):
        if not self.is_offboard_running:
            # 更新类成员变量中的默认坐标
            if default_x is not None: self._default_hover_x = default_x
            if default_y is not None: self._default_hover_y = default_y
            if default_z is not None: self._default_hover_z = default_z

            self.is_offboard_running = True
            self.offboard_task = asyncio.create_task(self._offboard_maintain_coroutine())
            logger.info("OFFBOARD Asynchronous keep-alive has been started")

    # 停止 OFFBOARD 保活
    def stop_offboard_maintain(self):
        self.is_offboard_running = False
        if self.offboard_task:
            self.offboard_task.cancel()
            logger.info("OFFBOARD Asynchronous keep-alive has been stopped")

    # 发送CTBR控制指令
    def set_ctbr_parameters_send(self, body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0, thrust=0.0):
        if self.is_offboard_running:
            logger.info("Detected the OFFBOARD holdover function is open. Automatically shutting down...")
            self.stop_offboard_maintain()
        # --- 改动 4: 发送前记录时间 ---
        local_time = time.time()
        self.master.mav.set_attitude_target_send(
            0,
            self.master.target_system,
            self.master.target_component,
            16,  # type_mask：仅启用「角速率+推力」，忽略姿态四元数
            [0.0, 0.0, 0.0, 0.0],  # q：四元数（被掩码忽略）
            body_roll_rate,
            body_pitch_rate,
            body_yaw_rate,
            thrust
        )
        # --- 改动 5: 发送后推送到同步中心 ---
        if self.data_sync:
            action = ActionData(
                time_sent_local=local_time,
                body_roll_rate=body_roll_rate,
                body_pitch_rate=body_pitch_rate,
                body_yaw_rate=body_yaw_rate,
                thrust=thrust
            )
            self.data_sync.on_new_action(action)
    
    # 发送CTBR控制指令（连续）
    def set_ctbr_parameters_continuously(self, body_roll_rate=[0.0, 0.0], body_pitch_rate=[0.0, 0.0], body_yaw_rate=[0.0, 0.0], thrust=[0.0, 0.0], frequency=20):
        for roll, pitch, yaw, t in zip(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust):
            self.set_ctbr_parameters_send(roll, pitch, yaw, t)
            time.sleep(1.0 / frequency)
    
    # 发送CTBR控制指令（重复）
    def set_ctbr_parameters_repeatly(self, body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0, thrust=0.0, frequency=20, repeat_times=10):
        for _ in range(repeat_times):
            self.set_ctbr_parameters_send(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust)
            time.sleep(1.0 / frequency)

    # ==============================================
    #  新增代码块：移植自第二段代码的功能
    # ==============================================
    def request_message_stream(self, msg_id, freq_hz=10):
        """请求飞控以指定频率发送特定消息"""
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,  # confirmation
            msg_id,  # param1: 消息 ID
            1e6 / freq_hz,  # param2: 间隔（微秒），例如 10Hz -> 100000us
            0, 0, 0, 0, 0  # 其他参数留空
        )

    def _recv_data_loop(self):
        """(线程内部) 接收数据并打印，逻辑完全照搬第二段代码"""
        while self.is_monitoring:
            # 核心逻辑：不指定 type，一次只取一条消息
            msg = self.master.recv_match(blocking=False)
            
            if not msg:
                time.sleep(0.001)
                continue

            # 根据消息 ID 或类型进行分发处理
            msg_type = msg.get_type()

            # ==============================================
            #  所有数据绑定 PX4 飞控启动时间片 time_boot_ms
            #  (以下打印逻辑完全保留你提供的第二段代码的原样)
            # ==============================================
            if msg_type == 'ATTITUDE':
                # 姿态消息：time_boot_ms (毫秒)
                px4_time = msg.time_boot_ms
                logger.info(f"[PX4时间: {px4_time:>8}ms] [角速率] Roll: {msg.rollspeed:+.4f} | Pitch: {msg.pitchspeed:+.4f} | Yaw: {msg.yawspeed:+.4f} rad/s")
            
            elif msg_type == 'ACTUATOR_OUTPUT_STATUS':
                # 电机消息：time_usec (微秒) → 转毫秒
                px4_time = msg.time_usec // 1000
                motors = msg.actuator[:4]
                logger.info(f"[PX4时间: {px4_time:>8}ms] [执行器] 电机: {[f'{m:.1f}' for m in motors]}")

            elif msg_type == 'LOCAL_POSITION_NED':
                # 位置消息：time_boot_ms (毫秒)
                px4_time = msg.time_boot_ms
                x = msg.x
                y = msg.y
                relative_alt = -msg.z
                logger.info(f"[PX4时间: {px4_time:>8}ms] [📌 坐标] X(前): {x:.2f} | Y(右): {y:.2f} | 高度: {relative_alt:.2f} m")

            # --- 改动 3: 新增数据推送到同步中心 ---
            if self.data_sync:
                self.data_sync.on_new_observation(msg)

    def start_monitoring(self, message_ids=[30, 375, 32], freq_hz=20):
        """启动数据监听线程"""
        if not self.is_monitoring:
            # 请求飞控发送数据流
            for msg_id in message_ids:
                self.request_message_stream(msg_id, freq_hz)
            logger.info("📡 已请求数据流")

            logger.info("🧹 正在清空 MAVLink 旧数据缓冲区...")
            start_flush_time = time.time()
            flushed_packets = 0
            
            # 循环清空策略：持续清空直到至少 0.2 秒内没有新数据
            # 或者最多清空 1 秒防止死循环
            last_msg_time = time.time()
            while True:
                msg = self.master.recv_match(blocking=False)
                if msg:
                    flushed_packets += 1
                    last_msg_time = time.time()
                else:
                    # 如果超过 0.2 秒没收到消息，认为缓冲区空了
                    if time.time() - last_msg_time > 0.2:
                        break
                    # 防止 CPU 100%
                    time.sleep(0.01)
                
                # 超时保护
                if time.time() - start_flush_time > 1.0:
                    break
            
            logger.info(f"🧹 缓冲区清空完成，丢弃了 {flushed_packets} 条旧消息")
            
            # 启动线程
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._recv_data_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """停止数据监听线程"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
            logger.info("🛑 数据监听已停止")
            self.stop_logging()

    # ==============================================
    #  新增代码块：发送线程功能
    # ==============================================
    def _send_ctbr_control_loop(self):
        """线程内部函数：循环读取 current_params 并发送"""
        logger.info(f"发送线程启动，频率: {self.send_frequency}Hz")
        while self.is_sending:
            # 1. 线程安全地读取当前参数
            with self.param_lock:
                roll = self.current_params.body_roll_rate
                pitch = self.current_params.body_pitch_rate
                yaw = self.current_params.body_yaw_rate
                thrust = self.current_params.thrust
            
            # 2. 调用你原有的函数发送
            self.set_ctbr_parameters_send(roll, pitch, yaw, thrust)
            
            # 3. 维持频率
            time.sleep(1.0 / self.send_frequency)
        logger.info("发送线程已停止")

    def start_ctbr_send_thread(self, frequency=50):
        """启动发送线程"""
        if not self.is_sending:
            self.send_frequency = frequency
            self.is_sending = True
            self.send_thread = threading.Thread(target=self._send_ctbr_control_loop, daemon=True)
            self.send_thread.start()

    def stop_ctbr_send_thread(self):
        """停止发送线程"""
        if self.is_sending:
            self.is_sending = False
            if self.send_thread:
                self.send_thread.join()

    def update_ctbr_send_params(self, body_roll_rate=None, body_pitch_rate=None, body_yaw_rate=None, thrust=None):
        with self.param_lock:
            if body_roll_rate is not None:
                self.current_params.body_roll_rate = body_roll_rate
            if body_pitch_rate is not None:
                self.current_params.body_pitch_rate = body_pitch_rate
            if body_yaw_rate is not None:
                self.current_params.body_yaw_rate = body_yaw_rate
            if thrust is not None:
                self.current_params.thrust = thrust