# Train PaddleOCR


训练PP-DocLayoutV3为例



```bash
# 安装基础依赖
uv sync

# windows?
winget install BYVoid.OpenCC --source winget

# 安装最新的paddlepaddle-gpu，我这里cuda13，命令如下
uv pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# paddlex安装模型等 PP-DocLayoutV3
uv run paddlex --install PaddleDetection

# 其他模型
uv run paddlex --install PaddleOCR
```












## 参考资料
[安装paddlepaddle-gpu命令](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)
