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
                use_sync_queue: bool = True, 
                enable_logging: bool = True):
        self.master = mavutil.mavlink_connection(connection_str, timeout=timeout)
        logger.info("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        logger.info(f"Connected to PX4 (system {self.master.target_system}, component {self.master.target_component})")

        self.is_offboard_running = False
        self.offboard_task = None
        self.is_monitor_ctbr_status = False
        self.is_monitor_ctbr_test_status = False

        self.thrust_output_index = thrust_output_index

        # --- 新增：数据监听相关状态 ---
        self.is_monitoring = False
        self.monitor_thread = None

        # --- 新增：发送线程相关状态 ---
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
                use_queue=use_sync_queue,
                logger=self._logger # 传入 logger
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
        """
        获取仿真时间管理器实例
        """
        if not self.data_sync:
            raise RuntimeError("必须开启 enable_data_sync=True 才能使用 SimTimeKeeper")
        return SimTimeKeeper(self.data_sync)

    # OFFBOARD 保活指令
    def send_hover_setpoint(self, x=0, y=0, z=-2.5):
        self.master.mav.set_attitude_target_send(
            0, self.master.target_system, self.master.target_component,
            16,
            [0.0, 0.0, 0.0, 0.0],
            0.0, 0.0, 0.0, 0.55
        )
        
    # 改变控制模式
    def change_control_mode(self, mode=6, is_maintain_offboard=False):
        if mode == 6:
            for _ in range(40):
                self.send_hover_setpoint()
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
            self.start_offboard_maintain()

    # OFFBOARD 异步保活协程
    async def _offboard_maintain_coroutine(self):
        try:
            while self.is_offboard_running:
                self.send_hover_setpoint()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logger.info("OFFBOARD keep-alive task cancelled gracefully")
        finally:
            logger.info("OFFBOARD maintain coroutine exited")

    # 启动 OFFBOARD 保活
    def start_offboard_maintain(self):
        if not self.is_offboard_running:
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