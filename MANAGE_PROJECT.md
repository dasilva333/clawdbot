# Managing MiniCPM-V-CookBook WebRTC Demo

This project is set up to run locally on your Mac Mini with Metal acceleration.

## 🚀 Quick Start (Recommended)

I've created a helper script `run_demo.sh` that handles all the complex environment setup for you.

To start all services:

```bash
cd ~/Projects/minicpm/demo/web_demo/WebRTC_Demo
./run_demo.sh start
```

## 🛑 Stopping Services

To stop all running services:

```bash
./run_demo.sh stop
```

## 📊 Status & Logs

Check if services are running:

```bash
./run_demo.sh status
```

View logs:

```bash
./run_demo.sh logs
```

## ⚙️ Configuration

- **Mode**: Default is **Simplex** (Voice-optimized).
- **Models**: Using optimized `Q4_K_M` quantization (~5GB).
- **Paths**: Python environment and dependencies are handled by `run_demo.sh`.

## 🌐 Access

- **Web Interface**: [https://localhost:8088](https://localhost:8088)
  - *Accept the security warning for the self-signed certificate.*
