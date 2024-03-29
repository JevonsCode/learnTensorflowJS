import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data";

window.onload = async () => {
    const data = getData(200, 3);

    tfvis.render.scatterplot(
        { name: "OVERFIT" },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 10,
        activation: "tanh",
        inputShape: [2],
        // 权重衰减
        kernelRegularizer: tf.regularizers.l2({ l2: 1 }),
    }));
    // 设置丢弃率 让模型变简单 丢弃 9/10 的模型
    model.add(tf.layers.dropout({ rate: 0.65 }));
    model.add(tf.layers.dense({
        units: 1,
        activation: "sigmoid",
    }));

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        validationSplit: 0.2,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            { name: " 训练过程" },
            ["loss", "val_loss"],
            { callbacks: ["onEpochEnd"] }
        )
    })
}