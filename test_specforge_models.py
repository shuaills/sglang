#!/usr/bin/env python3
"""
批量测试 SpecForge 输出的不同大小的 EAGLE3 模型
测试不同 hidden size 的 spec 模型的 acceptance length 性能
"""

import os
import sys
import time
import json
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# 配置
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SPECFORGE_OUTPUT_DIR = "/sgl-workspace/SpecForge/outputs"
BENCHMARK_DIR = "/sgl-workspace/sglang/benchmark/mtbench"
HOST = "0.0.0.0"
PORT = 30000
NUM_QUESTIONS = 80
PARALLEL = 1

# 可用模型列表（按照模型大小排序）
MODELS = [
    {
        "name": "tiny",
        "path": "eagle3-tiny-1.0data-projectionemb/epoch_1",
        "description": "Tiny model (512 hidden size)"
    },
    {
        "name": "small", 
        "path": "eagle3-small-1.0data-projectionemb/epoch_1",
        "description": "Small model"
    },
    {
        "name": "medium",
        "path": "eagle3-medium-1.0data-projectionemb/epoch_1", 
        "description": "Medium model (2048 hidden size)"
    },
    {
        "name": "large",
        "path": "eagle3-large-1.0data-projectionemb/epoch_1",
        "description": "Large model"
    },
    {
        "name": "pro",
        "path": "eagle3-pro-1.0data-projectionemb/epoch_1",
        "description": "Pro model"
    },
    {
        "name": "original",
        "path": "eagle3-original-1.0data-projectionemb/epoch_1",
        "description": "Original model"
    },
    {
        "name": "2layer",
        "path": "eagle3-2layer-1.0data-projectionemb/epoch_1",
        "description": "2-layer model"
    }
]

# 服务器启动参数
SERVER_ARGS = [
    "--speculative-algorithm", "EAGLE3",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1", 
    "--speculative-num-draft-tokens", "4",
    "--mem-fraction-static", "0.75",
    "--cuda-graph-max-bs", "2",
    "--tp", "2",
    "--context-length", "8192",
    "--trust-remote-code",
    "--host", HOST,
    "--port", str(PORT),
    "--dtype", "float16"
]

class ModelTester:
    def __init__(self):
        self.server_process = None
        self.results = []
        self.num_questions = NUM_QUESTIONS
        self.port = PORT
        self.server_args = SERVER_ARGS
        
    def check_model_exists(self, model_path):
        """检查模型路径是否存在"""
        full_path = os.path.join(SPECFORGE_OUTPUT_DIR, model_path)
        if not os.path.exists(full_path):
            print(f"❌ Model not found: {full_path}")
            return False
        
        config_path = os.path.join(full_path, "config.json")
        if not os.path.exists(config_path):
            print(f"❌ Config not found: {config_path}")
            return False
            
        return True
    
    def kill_existing_servers(self):
        """杀死现有的 sglang 服务器进程"""
        try:
            subprocess.run(["pkill", "-f", "sglang.launch_server"], 
                         capture_output=True, check=False)
            time.sleep(2)  # 等待进程完全关闭
        except:
            pass
    
    def start_server(self, model_name, model_path):
        """启动 sglang 服务器"""
        print(f"🚀 Starting server for {model_name}...")
        
        full_model_path = os.path.join(SPECFORGE_OUTPUT_DIR, model_path)
        
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model", BASE_MODEL,
            "--speculative-draft-model-path", full_model_path
        ] + self.server_args
        
        print(f"Command: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=os.setsid  # 创建新进程组
            )
            
            # 等待服务器启动
            print("⏳ Waiting for server to start...")
            for i in range(60):  # 最多等待60秒
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    print(f"❌ Server failed to start:")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    return False
                    
                # 检查服务器是否响应
                try:
                    import requests
                    response = requests.get(f"http://{HOST}:{self.port}/health", timeout=1)
                    if response.status_code == 200:
                        print(f"✅ Server started successfully for {model_name}")
                        return True
                except:
                    pass
                    
                time.sleep(1)
                print(f"   Waiting... ({i+1}/60)")
            
            print("❌ Server startup timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """停止服务器"""
        if self.server_process:
            try:
                # 杀死整个进程组
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=10)
            except:
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except:
                    pass
            self.server_process = None
        
        # 确保清理残留进程
        self.kill_existing_servers()
        time.sleep(2)
    
    def run_benchmark(self, model_name):
        """运行基准测试"""
        print(f"🏃 Running benchmark for {model_name}...")
        
        cmd = [
            sys.executable, "bench_sglang_eagle.py",
            "--num-questions", str(self.num_questions),
            "--parallel", str(PARALLEL),
            "--host", f"http://127.0.0.1",
            "--port", str(self.port),
            "--backend", "srt"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=BENCHMARK_DIR,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                print(f"✅ Benchmark completed for {model_name}")
                return result.stdout, result.stderr
            else:
                print(f"❌ Benchmark failed for {model_name}")
                print(f"STDERR: {result.stderr}")
                return None, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timeout for {model_name}")
            return None, "Timeout"
        except Exception as e:
            print(f"❌ Benchmark error for {model_name}: {e}")
            return None, str(e)
    
    def extract_metrics(self, output):
        """从输出中提取关键指标"""
        if not output:
            return {}
        
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            # 查找接收长度相关指标
            if "accept" in line.lower() or "acceptance" in line.lower():
                metrics["acceptance_info"] = line.strip()
            if "tokens/s" in line.lower():
                metrics["throughput"] = line.strip()
            if "latency" in line.lower():
                metrics["latency"] = line.strip()
        
        return metrics
    
    def test_model(self, model_info):
        """测试单个模型"""
        model_name = model_info["name"]
        model_path = model_info["path"]
        description = model_info["description"]
        
        print(f"\n{'='*60}")
        print(f"🔍 Testing {model_name}: {description}")
        print(f"   Path: {model_path}")
        print(f"{'='*60}")
        
        # 检查模型是否存在
        if not self.check_model_exists(model_path):
            result = {
                "model": model_name,
                "description": description,
                "status": "failed",
                "error": "Model not found",
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
            return
        
        # 启动服务器
        if not self.start_server(model_name, model_path):
            result = {
                "model": model_name,
                "description": description, 
                "status": "failed",
                "error": "Server startup failed",
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
            return
        
        try:
            # 运行基准测试
            stdout, stderr = self.run_benchmark(model_name)
            
            if stdout:
                metrics = self.extract_metrics(stdout)
                result = {
                    "model": model_name,
                    "description": description,
                    "status": "success",
                    "metrics": metrics,
                    "full_output": stdout,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {
                    "model": model_name,
                    "description": description,
                    "status": "failed", 
                    "error": stderr or "Unknown error",
                    "timestamp": datetime.now().isoformat()
                }
            
            self.results.append(result)
            
        finally:
            # 停止服务器
            print(f"🛑 Stopping server for {model_name}...")
            self.stop_server()
            time.sleep(3)  # 等待资源释放
    
    def save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"specforge_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Results saved to: {filename}")
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("📋 TEST SUMMARY")
        print(f"{'='*60}")
        
        for result in self.results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            print(f"{status_emoji} {result['model']:12} - {result['description']}")
            if result["status"] == "success" and "metrics" in result:
                for key, value in result["metrics"].items():
                    print(f"     {key}: {value}")
            elif result["status"] == "failed":
                print(f"     Error: {result.get('error', 'Unknown')}")
    
    def run_all_tests(self, model_filter=None):
        """运行所有测试"""
        print(f"🚀 Starting SpecForge model testing")
        print(f"⚙️  Configuration:")
        print(f"   Base model: {BASE_MODEL}")
        print(f"   Questions: {self.num_questions}")
        print(f"   Parallel: {PARALLEL}")
        print(f"   Port: {self.port}")
        
        # 清理现有进程
        self.kill_existing_servers()
        
        models_to_test = MODELS
        if model_filter:
            models_to_test = [m for m in MODELS if model_filter.lower() in m["name"].lower()]
            print(f"🔍 Filtering models with: {model_filter}")
        
        print(f"📋 Testing {len(models_to_test)} models:")
        for model in models_to_test:
            print(f"   - {model['name']}: {model['description']}")
        
        try:
            for i, model_info in enumerate(models_to_test, 1):
                print(f"\n🏃 Progress: {i}/{len(models_to_test)}")
                self.test_model(model_info)
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Testing interrupted by user")
        
        finally:
            self.stop_server()
            self.save_results()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SpecForge EAGLE3 models")
    parser.add_argument("--filter", help="Filter models by name (case insensitive)")
    parser.add_argument("--questions", type=int, default=80, 
                       help="Number of questions to test (default: 80)")
    parser.add_argument("--port", type=int, default=30000,
                       help="Server port (default: 30000)")
    
    args = parser.parse_args()
    
    # 使用局部变量而不是修改全局变量
    num_questions = args.questions
    port = args.port
    
    # 创建修改后的服务器参数
    server_args = SERVER_ARGS.copy()
    for i, arg in enumerate(server_args):
        if arg == str(PORT):
            server_args[i] = str(port)
    
    tester = ModelTester()
    tester.num_questions = num_questions
    tester.port = port
    tester.server_args = server_args
    tester.run_all_tests(model_filter=args.filter)

if __name__ == "__main__":
    main()