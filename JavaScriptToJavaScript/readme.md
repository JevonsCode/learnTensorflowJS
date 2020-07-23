## 分片

> [使用权重分片和量化将tfjs_layers_model转换为tfjs_layers_model](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#converting-tfjs_layers_model-to-tfjs_layers_model-with-weight-sharding-and-quantization)

```bash
tensorflowjs_converter --input_format=tfjs_layers_model --output_format=tfjs_layers_model --weight_shard _size_bytes=102400 data\mobilenet\web_model\model.json  data\mobilenet\sharded_model\
```

## 压缩

```bash
tensorflowjs_converter --input_format=tfjs_layers_model --output_format=tfjs_layers_model --quantization_bytes=2 data\mobilenet\web_model\model.json  data\mobilenet\quantize_model\
```

*精度会有一些影响*


## 加速

> [Converting tfjs_layers_model to tfjs_graph_model](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#converting-tfjs_layers_model-to-tfjs_graph_model)

```bash
tensorflowjs_converter --input_format=tfjs_layers_model --output_format=tfjs_graph_model --quantization_bytes=2 data\mobilenet\web_model\model.json  data\mobilenet\graph_model\
```