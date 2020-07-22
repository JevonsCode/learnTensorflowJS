# Python to JavaScript (Model)

## [环境](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)步骤

下载[miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)来管理 python 环境

> 注意是 Miniconda3-latest-Windows-x86_64.exe （这环境可是坑死我！）

```bash
# 用 conda 创建一个 python 3.6.8 的虚拟环境
conda create -n tfjs python=3.6.8
```

[安装 python 的 tensorflowjs](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#step-1-converting-a-tensorflow-savedmodel-tensorflow-hub-module-keras-hdf5-or-tfkeras-savedmodel-to-a-web-friendly-format)

```bash
pip install tensorflowjs
```

```bash
# pip 用清华的源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflowjs
```


## Python 转 JS

```bash
# 激活 tfjs 环境
conda activate tfjs
```

[转化要求](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#conversion-flags)

```bash
# 转换
tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model data\mobilenet\keras.h5 data\mobilenet\web_model_js\
```