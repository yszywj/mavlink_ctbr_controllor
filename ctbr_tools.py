import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Deque, List
from collections import deque
import time

# ==============================================
# 1. 数据类定义 (Data Classes)
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
    """最终对齐后的数据，供 RL 使用"""
    obs: ObservationData = field(default_factory=ObservationData)
    last_action: Optional[ActionData] = None

# ==============================================
# 2. 核心同步类 (DroneDataSync)
# ==============================================

class DroneDataSync:
    def __init__(self, 
                 use_condition: bool = True, 
                 use_queue: bool = True,
                 action_history_len: int = 10,
                 max_queue_size: int = 1000,
                 sync_window_ms: int = 50):
        """
        初始化数据同步中心
        :param use_condition: 是否启用 Condition 实时流 (推荐用于控制)
        :param use_queue: 是否启用 Queue 记录流 (推荐用于日志)
        :param action_history_len: 保留历史指令的长度
        :param max_queue_size: Queue 的最大长度
        """
        self._use_condition = use_condition
        self._use_queue = use_queue
        self._sync_window_ms = sync_window_ms

        # --- 共享状态 ---
        self._latest_px4_time_est: int = 0  # 最新的飞控时间估算
        
        # --- 1. Condition 机制 (实时控制) ---
        if self._use_condition:
            self._lock = threading.Lock()
            self._cv = threading.Condition(self._lock)
            self._has_new_data = False
            self._latest_obs: ObservationData = ObservationData()
            self._action_history: Deque[ActionData] = deque(maxlen=action_history_len)

            # 【新增】帧缓存：临时存放未凑齐的单条消息
            self._frame_cache = {
                'ATTITUDE': None,
                'LOCAL_POSITION_NED': None,
                'ACTUATOR_OUTPUT_STATUS': None
            }

        # --- 2. Queue 机制 (完整记录) ---
        if self._use_queue:
            self._obs_queue: queue.Queue[ObservationData] = queue.Queue(maxsize=max_queue_size)
            self._action_queue: queue.Queue[ActionData] = queue.Queue(maxsize=max_queue_size)

    # ==============================================
    #  写入接口 (由 CTBRController 调用)
    # ==============================================

    def on_new_observation(self, msg):
        """
        当收到 PX4 新消息时调用此函数
        :param msg: pymavlink 的消息对象
        """
        msg_type = msg.get_type()
        
        # 更新最新的飞控时间
        current_px4_time = 0
        if hasattr(msg, 'time_boot_ms'):
            current_px4_time = msg.time_boot_ms
        elif hasattr(msg, 'time_usec'):
            current_px4_time = msg.time_usec // 1000
        
        if current_px4_time > 0:
            self._latest_px4_time_est = current_px4_time

        # --- Condition 模式写入 (核心修改：先缓存，凑齐再通知) ---
        if self._use_condition:
            with self._lock:
                # 1. 先把当前消息存入缓存
                if msg_type in self._frame_cache:
                    self._frame_cache[msg_type] = msg
                
                # 2. 【核心】尝试凑齐一帧并发出
                self._try_emit_synced_frame()

        # --- Queue 模式写入 ---
        if self._use_queue:
            # 为了保证 Queue 里的数据是完整的快照，我们每次创建新对象
            # 注意：这里为了简单，我们只在收到特定消息时才入队，或者你可以选择每次都入队
            # 这里演示：只有收到 ATTITUDE 时才生成一个完整快照入队
            if msg_type == 'ATTITUDE':
                obs_snapshot = ObservationData()
                # 把当前 Condition 里的完整状态复制出来 (需要在锁外复制，或者优化这里)
                # 简化版：直接用当前消息更新一个新的
                obs_snapshot.time_boot_ms = current_px4_time
                # 注意：简单的 Queue 实现可能会导致数据只有部分更新，
                # 生产环境建议在这里维护一个专门给 Queue 用的状态机。
                # 此处为了保持代码简洁，我们主要依赖 Condition 里的逻辑，
                # Queue 仅做演示或用于对数据完整性要求不高的场景。
                try:
                    self._obs_queue.put_nowait(obs_snapshot)
                except queue.Full:
                    pass # 队列满了就丢弃，防止内存泄漏

    def on_new_action(self, action: ActionData):
        """
        当发送了新的控制指令时调用此函数
        :param action: ActionData 对象
        """
        # 补全飞控时间估算
        if action.time_boot_ms_est == 0:
            action.time_boot_ms_est = self._latest_px4_time_est

        # --- Condition 模式写入 ---
        if self._use_condition:
            with self._lock:
                self._action_history.append(action)

        # --- Queue 模式写入 ---
        if self._use_queue:
            try:
                self._action_queue.put_nowait(action)
            except queue.Full:
                pass

    # ==============================================
    #  读取接口 (由用户/RL 调用)
    # ==============================================

    # --- Condition 模式接口 ---
    def wait_for_synced_data(self, timeout: float = None) -> Optional[SyncedData]:
        """
        [Condition 模式] 阻塞等待新数据，并自动对齐
        :return: SyncedData (obs + last_action)
        """
        if not self._use_condition:
            return None

        with self._cv:
            # 等待通知
            signaled = self._cv.wait_for(lambda: self._has_new_data, timeout=timeout)
            if not signaled:
                return None # 超时
            
            self._has_new_data = False
            
            # 组装数据
            synced = SyncedData()
            synced.obs = ObservationData(**self._latest_obs.__dict__) # 深拷贝一份
            
            # 寻找最近的一条指令
            if self._action_history:
                # 简单策略：取最后一条
                # 高级策略：可以在这里基于时间戳做更精确的匹配
                synced.last_action = ActionData(**self._action_history[-1].__dict__)
            
            return synced

    # --- Queue 模式接口 ---
    def get_queued_obs(self, block: bool = False, timeout: float = 0.0) -> Optional[ObservationData]:
        """[Queue 模式] 获取观测数据"""
        if not self._use_queue: return None
        try:
            return self._obs_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_queued_action(self, block: bool = False, timeout: float = 0.0) -> Optional[ActionData]:
        """[Queue 模式] 获取指令数据"""
        if not self._use_queue: return None
        try:
            return self._action_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    # ==============================================
    #  内部辅助函数
    # ==============================================
    def _try_emit_synced_frame(self):
        """
        [线程安全] 检查缓存：如果3个消息都齐了且时间戳在同一窗口内，就更新数据并通知
        必须在持有 self._lock 的情况下调用！
        """
        cache = self._frame_cache
        
        # 1. 检查：3个消息是不是都收到了？
        if not (cache['ATTITUDE'] and cache['LOCAL_POSITION_NED'] and cache['ACTUATOR_OUTPUT_STATUS']):
            return # 没凑齐，直接返回，继续等
        
        # 2. 【修复】安全地获取3个时间戳
        # 先取姿态和位置的时间（这两个肯定有 time_boot_ms）
        t_att = cache['ATTITUDE'].time_boot_ms
        t_pos = cache['LOCAL_POSITION_NED'].time_boot_ms
        
        # 再处理电机时间（可能没有 time_boot_ms，没有就用姿态时间代替）
        t_act = t_att
        if hasattr(cache['ACTUATOR_OUTPUT_STATUS'], 'time_boot_ms'):
            t_act = cache['ACTUATOR_OUTPUT_STATUS'].time_boot_ms
        
        # 3. 检查时间差
        times = [t_att, t_pos, t_act]
        if max(times) - min(times) > self._sync_window_ms:
            return # 时间差太大，不是同一帧，继续等
        
        # 4. ✅ 凑齐了！合并成完整观测
        self._update_obs_with_msg(self._latest_obs, cache['ATTITUDE'], 'ATTITUDE')
        self._update_obs_with_msg(self._latest_obs, cache['LOCAL_POSITION_NED'], 'LOCAL_POSITION_NED')
        self._update_obs_with_msg(self._latest_obs, cache['ACTUATOR_OUTPUT_STATUS'], 'ACTUATOR_OUTPUT_STATUS')
        
        # 5. 【关键】只通知一次！
        self._has_new_data = True
        self._cv.notify()
        
        # 6. 清空缓存，准备接收下一帧
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

@dataclass
class ControlParams:
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.55