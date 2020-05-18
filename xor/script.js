/**
 * XOR 数据集：当数据都为正/负的时候 label 为 0
 */


import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data";

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: "XOR" },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();

    // 隐藏层
    model.add(tf.layers.dense({
        units: 4,
        inputShape: [2],
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 1,
        // 上一个为 4 个神经元，这一层就有 4 个 inputShape, 会默认添加
        activation: "sigmoid"
    }));

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map((r) => [r.x, r.y]));
    const labels = tf.tensor(data.map((r) => r.label));

    await model.fit(inputs, labels, {
        // batchSize: 4,
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: "训练过程" },
            ["loss"]
        )
    });


    document.getElementById("btn").disabled = false;

    window.predict = () => {
        const inputx = document.getElementById("inputx");
        const inputy = document.getElementById("inputy");
        console.log(inputx.value * 1, inputy.value * 1);
        const pred = model.predict(tf.tensor([[inputx.value * 1, inputy.value * 1]]));
        alert(pred.dataSync()[0]);
    }
}