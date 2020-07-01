/**
 * 
 * hs data --cors 启动 mobilenet 模型
 * 
 * parcel ./index.html 启动本项目
 */

import * as tf from "@tensorflow/tfjs";
import { img2x, file2img } from "./utils";

const MOBILENET_MODEL_PATH = "http://127.0.0.1:8080/mobilenet/web_model/model.json";
const BRAND_MODEL_PATH = "http://127.0.0.1:8080/brand/web_model/model.json";

const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {

    // loadLayersModel: Load a model composed of Layer objects, including its topology and optionally weights.
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

    // 查看一下模型结构，为了更好的思考在哪阶段  // Print a text summary of the model's layers.
    mobilenet.summary();


    // 从 conv_pw_13_relu  这一层截断，conv 是卷积的简写

    // Retrieves a layer based on either its name (unique) or index.
    const layer = mobilenet.getLayer("conv_pw_13_relu");

    // 把原 mobilenet 的输入当这个新建模型的输入，截断的层当作 outputs 完成截断
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });


    // 加载
    const model = await tf.loadLayersModel(BRAND_MODEL_PATH);

    window.predict = async (f) => {
        const img = await file2img(f);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const input = truncatedMobilenet.predict(img2x(img))
            return model.predict(input);
        });
        console.log("-> ", pred, pred.argMax(1).dataSync())
        // 第二维取 1
        const index = pred.argMax(1).dataSync()[0];
        console.log("结果", index, BRAND_CLASSES[index]);
    }
};