# Train PaddleOCR


训练PP-DocLayoutV3为例


## 1. 安装依赖
```shell
# 安装基础依赖
uv sync

# windows?
winget install BYVoid.OpenCC --source winget

# 安装最新的paddlepaddle-gpu，我这里cuda13，命令如下
uv pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# paddlex安装模型等 PP-DocLayoutV3
mkdir '.\.venv\Lib\site-packages\paddlex\repo_manager\repos'
uv run paddlex --install PaddleDetection

# 其他模型
uv run paddlex --install PaddleOCR



cp /.venv/lib/python/site-packages/paddlex/repo_manager/repos/PaddleDetection/configs/layout_analysis/PP-DocLayoutV3.yaml
```

## 2. 训练
```shell
$env:PATH = ";J:\ProjectFiles\train_paddleocr\.venv\Lib\site-packages\paddle\include\paddle\phi\backends\dynload;$env:PATH"
$env:PATH = ";J:\ProjectFiles\train_paddleocr\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"


python main.py --mode train --device gpu:0

python main.py --mode export --device gpu:0
```



## 参考资料
[安装paddlepaddle-gpu命令](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)
