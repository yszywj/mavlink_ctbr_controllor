from pymavlink import mavutil
import time
import asyncio
import os
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CTBRController")

class CTBRControllor:
    def __init__(self, connection_str='udp:0.0.0.0:14550', timeout=30, log_dir="./log_folder", thrust_output_index=None):
        self.master = mavutil.mavlink_connection(connection_str, timeout=timeout)
        logger.info("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        logger.info(f"Connected to PX4 (system {self.master.target_system}, component {self.master.target_component})")

        self.is_offboard_running = False
        self.offboard_task = None
        self.is_monitor_ctbr_status = False
        self.is_monitor_ctbr_test_status = False

        self.thrust_output_index = thrust_output_index

        # ====================== 新增：数据记录与日志系统初始化 ======================
        self.log_dir = log_dir
        self.current_log_subdir = None
        
        # 数据队列：用于在监控协程和写入协程之间传递数据
        self.data_queue = asyncio.Queue(maxsize=5000)
        
        # 写入协程控制
        self._writer_task = None
        self._is_writer_running = False
        
        # 数据缓存：用于对齐不同频率的传感器数据
        self._last_actuator_output = None  # 缓存上一次的推力数据
        self._last_attitude = None          # 缓存上一次的姿态数据
        
        # 批量写入缓冲 (减少IO次数)
        self._actual_buffer = []
        self._target_buffer = []
        self._buffer_flush_threshold = 100  # 每攒100条写一次
        # ============================================================================

    # OFFBOARD 保活指令
    def send_hover_setpoint(self, x=0, y=0, z=-2.5):
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            x, y, z,
            0,0,0, 0,0,0, 0,0
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
    def set_ctbr_parameters(self, body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0, thrust=0.0):
        if self.is_offboard_running:
            logger.info("Detected the OFFBOARD holdover function is open. Automatically shutting down...")
            self.stop_offboard_maintain()
        self.master.mav.set_attitude_target_send(
            0,
            self.master.target_system,
            self.master.target_component,
            16,  # type_mask：仅启用「角速率+推力」，忽略姿态四元数
            [1.0, 0.0, 0.0, 0.0],  # q：四元数（被掩码忽略）
            body_roll_rate,
            body_pitch_rate,
            body_yaw_rate,
            thrust
        )
    
    # 发送CTBR控制指令（连续）
    def set_ctbr_parameters_continuously(self, body_roll_rate=[0.0, 0.0], body_pitch_rate=[0.0, 0.0], body_yaw_rate=[0.0, 0.0], thrust=[0.0, 0.0], frequency=20):
        for roll, pitch, yaw, t in zip(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust):
            self.set_ctbr_parameters(roll, pitch, yaw, t)
            time.sleep(1.0 / frequency)
    
    # 发送CTBR控制指令（重复）
    def set_ctbr_parameters_repeatly(self, body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0, thrust=0.0, frequency=20, repeat_times=10):
        for _ in range(repeat_times):
            self.set_ctbr_parameters(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust)
            time.sleep(1.0 / frequency)

    def _init_logging_directory(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_log_subdir = os.path.join(self.log_dir, timestamp_str)
        os.makedirs(self.current_log_subdir, exist_ok=True)
        logger.info(f"Logging directory initialized at: {self.current_log_subdir}")

    async def _writer_coroutine(self):
        logger.info("Data Writer coroutine started.")
        try:
            while self._is_writer_running or not self.data_queue.empty():
                try:
                    # 尝试从队列获取数据，设置超时以便检查退出标志
                    data_type, data_point = await asyncio.wait_for(self.data_queue.get(), timeout=0.1)
                    
                    if data_type == 'actual':
                        self._actual_buffer.append(data_point)
                    elif data_type == 'target':
                        self._target_buffer.append(data_point)
                    
                    # 检查是否需要刷新到磁盘
                    if len(self._actual_buffer) >= self._buffer_flush_threshold:
                        self._flush_buffer('actual')
                    if len(self._target_buffer) >= self._buffer_flush_threshold:
                        self._flush_buffer('target')
                        
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.error(f"Writer coroutine error: {e}")
        finally:
            # 退出前确保所有缓冲数据都写入磁盘
            self._flush_buffer('actual', force=True)
            self._flush_buffer('target', force=True)
            logger.info("Data Writer coroutine stopped. All data flushed.")
    
    def _flush_buffer(self, data_type, force=False):
        """将内存缓冲写入 Parquet 文件"""
        buffer = self._actual_buffer if data_type == 'actual' else self._target_buffer

        if not buffer:
            return
            
        if not force and len(buffer) < self._buffer_flush_threshold:
            return

        filename = f"ctbr_{data_type}.parquet"
        filepath = os.path.join(self.current_log_subdir, filename)
        
        # 转换为 DataFrame
        df = pd.DataFrame(buffer)
        
        # 确保时间戳列为数值类型
        if 'time_boot_ms' in df.columns:
            df['time_boot_ms'] = pd.to_numeric(df['time_boot_ms'])

        # 写入 Parquet
        # 如果文件存在，则追加；不存在则创建
        try:
            if os.path.exists(filepath):
                # 读取旧文件，合并新数据，再写入 (为了性能，实际生产中建议用更复杂的追加逻辑)
                # 这里简化处理：每次 flush 写一个新的 row group
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(table, root_path=os.path.dirname(filepath), 
                                    basename_template=filename, 
                                    existing_data_behavior='overwrite_or_ignore')
            else:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, filepath)
            
            # 清空缓冲
            if data_type == 'actual':
                self._actual_buffer = []
            else:
                self._target_buffer = []
                
        except Exception as e:
            logger.error(f"Error writing to Parquet: {e}")

    # 监听 ATTITUDE 状态消息
    async def _ctbr_status_monitor(self, frequency=20):
        while self.is_monitor_ctbr_status:
            # 1. 非阻塞读取所有可能的消息，更新缓存
            # 注意：这里用循环尽可能把缓冲区的消息读完，防止积压
            while True:
                msg = self.master.recv_match(blocking=False)
                if not msg:
                    break
                
                msg_type = msg.get_type()
                
                if msg_type == 'ATTITUDE':
                    self._last_attitude = msg
                elif msg_type == 'ACTUATOR_OUTPUT_STATUS':
                    self._last_actuator_output = msg
                elif msg_type == 'ATTITUDE_TARGET' and self.is_monitor_ctbr_test_status:
                    # 处理 Target 数据
                    target_data = {
                        "time_boot_ms": getattr(msg, 'time_boot_ms', int(time.time() * 1000)),
                        "roll_rate": getattr(msg, 'rollspeed', 0.0),
                        "pitch_rate": getattr(msg, 'pitchspeed', 0.0),
                        "yaw_rate": getattr(msg, 'yawspeed', 0.0),
                        "thrust": getattr(msg, 'thrust', 0.0)
                    }
                    # 放入队列
                    if self._is_writer_running:
                        try:
                            self.data_queue.put_nowait(('target', target_data))
                        except asyncio.QueueFull:
                            pass # 队列满了就丢，保证飞控控制优先
                        
                    # 打印调试
                    logger.info(f"TARGET -> t:{target_data['time_boot_ms']} | r:{target_data['roll_rate']:.2f} | p:{target_data['pitch_rate']:.2f} | y:{target_data['yaw_rate']:.2f} | t:{target_data['thrust']:.2f}")

            # 2. 生成 Actual 数据 (状态对齐逻辑)
            # 只有当我们同时有了姿态和推力数据时，才生成一条完整记录
            if self._last_attitude and self._last_actuator_output:
                # 以姿态数据的时间戳为基准
                current_time_ms = self._last_attitude.time_boot_ms
                
                actual_data = {
                    "time_boot_ms": current_time_ms,
                    "roll_rate": self._last_attitude.rollspeed,
                    "pitch_rate": self._last_attitude.pitchspeed,
                    "yaw_rate": self._last_attitude.yawspeed,
                    "thrust": self._last_actuator_output.actuator_output[0] # 假设第一个是主推力
                }
                
                # 放入队列 (供外部RL训练读取) 和 写入缓冲
                if self._is_writer_running:
                    try:
                        self.data_queue.put_nowait(('actual', actual_data))
                    except asyncio.QueueFull:
                        pass

                # 打印调试
                if self.is_monitor_ctbr_test_status:
                    logger.info(f"ACTUAL -> t:{actual_data['time_boot_ms']} | r:{actual_data['roll_rate']:.2f} | p:{actual_data['pitch_rate']:.2f} | y:{actual_data['yaw_rate']:.2f} | t:{actual_data['thrust']:.2f}")

            await asyncio.sleep(1.0 / frequency)

    def start_monitor_ctbr_status(self, frequency=20):
        if not self.is_monitor_ctbr_status:
            # 1. 初始化日志目录
            self._init_logging_directory()
            
            # 2. 启动写入协程
            self._is_writer_running = True
            self._writer_task = asyncio.create_task(self._writer_coroutine())
            
            # 3. 启动监控
            self.is_monitor_ctbr_status = True
            asyncio.create_task(self._ctbr_status_monitor(frequency=frequency))
            logger.info("Started monitoring CTBR status.")

    async def stop_monitor_ctbr_status(self):
        self.is_monitor_ctbr_status = False
        
        # 停止写入协程并等待刷新
        self._is_writer_running = False
        if self._writer_task:
            await self._writer_task
            
        logger.info("Stopped monitoring CTBR status.")

    # 设置监听 ATTITUDE 状态消息（测试模式）
    def set_monitor_ctbr_status_test_mode(self, enable: bool):
        self.is_monitor_ctbr_test_status = enable
        logger.info(f"test mode: {'open' if enable else 'closed'}")