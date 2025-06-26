#!/bin/bash

# IndexTTS API 服务启动脚本

# 默认配置
DEFAULT_PORT=8000
DEFAULT_HOST="127.0.0.1"

# 从命令行参数或环境变量获取配置
PORT=${1:-${API_PORT:-$DEFAULT_PORT}}
HOST=${API_HOST:-$DEFAULT_HOST}

# 显示使用说明
show_usage() {
    echo "使用方法："
    echo "  $0 [端口号]"
    echo "  $0 8080                    # 在端口 8080 启动服务"
    echo ""
    echo "环境变量："
    echo "  API_PORT=8080 $0          # 通过环境变量设置端口"
    echo "  API_HOST=127.0.0.1 $0     # 通过环境变量设置主机"
    echo ""
    echo "默认配置："
    echo "  端口: $DEFAULT_PORT"
    echo "  主机: $DEFAULT_HOST"
    exit 0
}

# 检查帮助参数
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
fi

echo "正在启动 IndexTTS API 服务..."
echo "配置信息："
echo "  主机: $HOST"
echo "  端口: $PORT"
echo ""

# 检查是否安装了必要的依赖
echo "检查 Python 依赖..."
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的依赖包。请运行: pip install fastapi uvicorn"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "checkpoints/config.yaml" ]; then
    echo "错误: 找不到模型配置文件 checkpoints/config.yaml"
    echo "请确保模型文件已正确放置在 checkpoints 目录中"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动服务
echo "启动服务 (http://$HOST:$PORT)..."
echo "API 文档: http://localhost:$PORT/docs"
echo "Swagger UI: http://localhost:$PORT/docs"
echo "ReDoc: http://localhost:$PORT/redoc"
echo "按 Ctrl+C 停止服务"
echo ""

# 使用 uvicorn 启动，支持自定义端口和主机
uvicorn api_server:app --host "$HOST" --port "$PORT" --workers 1 