import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data";

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: "逻辑回归训练数据" },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        },
        {
            xAxisDomain: [-6, 6],
            yAxisDomain: [-6, 6]
        }
    );

    // const model = tf.sequential();
    // model.add(tf.layers.dense({
    //     units: 1,
    //     inputShape: [2],
    //     // 激活函数  sigmoid 为了把值压缩到 0~1 之间
    //     activation: "sigmoid"
    // }));

    // model.compile({
    //     loss: tf.losses.logLoss,
    //     // adam 可以帮你自动调节学习率
    //     optimizer: tf.train.adam(0.1)
    // });

    // const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    // const labels = tf.tensor(data.map(p => p.label));

    // await model.fit(inputs, labels, {
    //     // 一轮几个
    //     batchSize: 40,
    //     // 训练几轮
    //     epochs: 100,
    //     callbacks: tfvis.show.fitCallbacks(
    //         { name: "训练过程" },
    //         ["loss"]
    //     )
    // });

    // document.getElementById("btn").disabled = false;

    // window.predict = () => {
    //     const inputx = document.getElementById("inputx");
    //     const inputy = document.getElementById("inputy");
    //     console.log(inputx.value * 1, inputy.value * 1);
    //     const pred = model.predict(tf.tensor([[inputx.value * 1, inputy.value * 1]]));
    //     alert(pred.dataSync()[0]);
    // }

}