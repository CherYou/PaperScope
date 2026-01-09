#!/usr/bin/env python3
"""
性能监控模块
用于监控随机游走算法的性能指标，包括内存使用、执行时间、CPU使用率等
"""

import time
import psutil
import os
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    execution_time: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval: float = 1.0):
        """
        初始化性能监控器
        
        Args:
            monitor_interval: 监控间隔（秒）
        """
        self.monitor_interval = monitor_interval
        self.process = psutil.Process(os.getpid())
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = None
        self.phase_metrics = defaultdict(list)
        self.current_phase = "initialization"
        
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("性能监控已停止")
    
    def set_phase(self, phase_name: str):
        """设置当前执行阶段"""
        self.current_phase = phase_name
        print(f"进入阶段: {phase_name}")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 获取当前指标
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                current_time = time.time()
                execution_time = current_time - self.start_time if self.start_time else 0
                
                metrics = PerformanceMetrics(
                    timestamp=current_time,
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    execution_time=execution_time
                )
                
                self.metrics_history.append(metrics)
                self.phase_metrics[self.current_phase].append(metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def add_custom_metric(self, name: str, value: float):
        """添加自定义指标"""
        if self.metrics_history:
            self.metrics_history[-1].custom_metrics[name] = value
    
    def get_current_memory(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_peak_memory(self) -> float:
        """获取峰值内存使用量（MB）"""
        if not self.metrics_history:
            return 0.0
        return max(m.memory_mb for m in self.metrics_history)
    
    def get_average_cpu(self) -> float:
        """获取平均CPU使用率"""
        if not self.metrics_history:
            return 0.0
        return sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
    
    def get_phase_summary(self) -> Dict[str, Dict[str, float]]:
        """获取各阶段性能摘要"""
        summary = {}
        
        for phase, metrics in self.phase_metrics.items():
            if not metrics:
                continue
                
            memory_values = [m.memory_mb for m in metrics]
            cpu_values = [m.cpu_percent for m in metrics]
            
            summary[phase] = {
                'duration': metrics[-1].execution_time - metrics[0].execution_time if len(metrics) > 1 else 0,
                'peak_memory_mb': max(memory_values),
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'sample_count': len(metrics)
            }
        
        return summary
    
    def save_metrics(self, output_path: str):
        """保存性能指标到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换为可序列化的格式
        metrics_data = []
        for metric in self.metrics_history:
            data = {
                'timestamp': metric.timestamp,
                'memory_mb': metric.memory_mb,
                'cpu_percent': metric.cpu_percent,
                'execution_time': metric.execution_time,
                'custom_metrics': metric.custom_metrics
            }
            metrics_data.append(data)
        
        # 保存详细指标
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics_data,
                'phase_summary': self.get_phase_summary(),
                'total_duration': self.get_total_duration(),
                'peak_memory': self.get_peak_memory(),
                'average_cpu': self.get_average_cpu()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"性能指标已保存到: {output_path}")
    
    def get_total_duration(self) -> float:
        """获取总执行时间"""
        if not self.metrics_history:
            return 0.0
        return self.metrics_history[-1].execution_time
    
    def plot_metrics(self, output_dir: str = "output/performance"):
        """绘制性能指标图表"""
        if not self.metrics_history:
            print("没有性能数据可绘制")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        timestamps = [m.timestamp for m in self.metrics_history]
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_percent for m in self.metrics_history]
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('random walk algorithm peformance monitor', fontsize=16)
        
        # 内存使用趋势
        ax1.plot(execution_times, memory_usage, 'b-', linewidth=2)
        ax1.set_title('memory usage trend')
        ax1.set_xlabel('execution time (s)')
        ax1.set_ylabel('memory usage (MB)')
        ax1.grid(True, alpha=0.3)
        
        # CPU使用率
        ax2.plot(execution_times, cpu_usage, 'r-', linewidth=2)
        ax2.set_title('cpu usage trend')
        ax2.set_xlabel('execution time (s)')
        ax2.set_ylabel('cpu usage (%)')
        ax2.grid(True, alpha=0.3)
        
        # 各阶段内存使用对比
        phase_summary = self.get_phase_summary()
        phases = list(phase_summary.keys())
        peak_memories = [phase_summary[phase]['peak_memory_mb'] for phase in phases]
        
        ax3.bar(phases, peak_memories, color='skyblue', alpha=0.7)
        ax3.set_title('peak memory usage of each phase')
        ax3.set_ylabel('peak memory (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 各阶段执行时间
        durations = [phase_summary[phase]['duration'] for phase in phases]
        ax4.bar(phases, durations, color='lightgreen', alpha=0.7)
        ax4.set_title('execution time of each phase')
        ax4.set_ylabel('execution time (s)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"性能图表已保存到: {plot_path}")
        
        # 创建详细的阶段分析图
        self._plot_phase_details(output_dir)
    
    def _plot_phase_details(self, output_dir: str):
        """绘制详细的阶段分析图"""
        if not self.phase_metrics:
            return
        
        fig, axes = plt.subplots(len(self.phase_metrics), 2, 
                                figsize=(15, 4 * len(self.phase_metrics)))
        
        if len(self.phase_metrics) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (phase, metrics) in enumerate(self.phase_metrics.items()):
            if not metrics:
                continue
            
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_mb for m in metrics]
            cpu_usage = [m.cpu_percent for m in metrics]
            
            # 内存使用
            axes[i, 0].plot(execution_times, memory_usage, 'b-', linewidth=2)
            axes[i, 0].set_title(f'{phase} - memory usage')
            axes[i, 0].set_xlabel('execution time (s)')
            axes[i, 0].set_ylabel('memory usage (MB)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # CPU使用率
            axes[i, 1].plot(execution_times, cpu_usage, 'r-', linewidth=2)
            axes[i, 1].set_title(f'{phase} - cpu usage')
            axes[i, 1].set_xlabel('execution time (s)')
            axes[i, 1].set_ylabel('cpu usage (%)')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        phase_plot_path = os.path.join(output_dir, 'phase_details.png')
        plt.savefig(phase_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"阶段详细图表已保存到: {phase_plot_path}")
    
    def print_summary(self):
        """打印性能摘要"""
        if not self.metrics_history:
            print("没有性能数据")
            return
        
        print(f"\n=== 性能监控摘要 ===")
        print(f"总执行时间: {self.get_total_duration():.2f} 秒")
        print(f"峰值内存使用: {self.get_peak_memory():.2f} MB")
        print(f"平均CPU使用率: {self.get_average_cpu():.2f}%")
        print(f"监控样本数: {len(self.metrics_history)}")
        
        # 各阶段摘要
        phase_summary = self.get_phase_summary()
        if phase_summary:
            print(f"\n=== 各阶段性能 ===")
            for phase, stats in phase_summary.items():
                print(f"{phase}:")
                print(f"  执行时间: {stats['duration']:.2f} 秒")
                print(f"  峰值内存: {stats['peak_memory_mb']:.2f} MB")
                print(f"  平均内存: {stats['avg_memory_mb']:.2f} MB")
                print(f"  平均CPU: {stats['avg_cpu_percent']:.2f}%")


class TimedContext:
    """计时上下文管理器"""
    
    def __init__(self, monitor: PerformanceMonitor, phase_name: str):
        self.monitor = monitor
        self.phase_name = phase_name
        self.start_time = None
    
    def __enter__(self):
        self.monitor.set_phase(self.phase_name)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.add_custom_metric(f"{self.phase_name}_duration", duration)
        print(f"{self.phase_name} 完成，用时: {duration:.2f} 秒")


def benchmark_function(func: Callable, *args, monitor: Optional[PerformanceMonitor] = None, 
                      phase_name: str = "benchmark", **kwargs):
    """
    对函数进行性能基准测试
    
    Args:
        func: 要测试的函数
        *args: 函数参数
        monitor: 性能监控器
        phase_name: 阶段名称
        **kwargs: 函数关键字参数
        
    Returns:
        函数返回值和执行时间
    """
    if monitor:
        with TimedContext(monitor, phase_name):
            start_memory = monitor.get_current_memory()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = monitor.get_current_memory()
            
            monitor.add_custom_metric(f"{phase_name}_memory_delta", end_memory - start_memory)
            
            return result, end_time - start_time
    else:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return result, end_time - start_time


if __name__ == "__main__":
    # 测试性能监控器
    monitor = PerformanceMonitor(monitor_interval=0.5)
    
    try:
        monitor.start_monitoring()
        
        # 模拟不同阶段的工作
        with TimedContext(monitor, "initialization"):
            time.sleep(2)
        
        with TimedContext(monitor, "data_processing"):
            # 模拟内存密集型操作
            data = [i for i in range(1000000)]
            time.sleep(3)
        
        with TimedContext(monitor, "computation"):
            # 模拟CPU密集型操作
            result = sum(i * i for i in range(100000))
            time.sleep(2)
        
    finally:
        monitor.stop_monitoring()
        monitor.print_summary()
        monitor.save_metrics("test_performance.json")
        monitor.plot_metrics("test_output")