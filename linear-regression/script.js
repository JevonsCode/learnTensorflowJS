import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
const fn = (x) => {
    const arr = [];
    let n = x;
    while (n < 10 * x) {
        arr.push(n);
        n += x;
    }
    return arr
}

window.onload = async () => {
    const xs = fn(1);
    const ys = fn(3);

    tfvis.render.scatterplot({
        name: "线性回归"
    }, {
        values: xs.map((x, index) => ({ x, y: ys[index] }))
    }, {
        xAxisDomain: [0, 10],
        yAxisDomain: [0, 50]
    });

    const model = tf.sequential();
    // units 神经元  inputShape 
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
        // 损失 (均方误差)
        loss: tf.losses.meanSquaredError,
        // 降低损失
        optimizer: tf.train.sgd(0.001)
    });

    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);
    // 拟合
    await model.fit(inputs, labels, {
        // 模型每次要学的样本数量
        batchSize: 10,
        // 迭代整个训练的次数
        epochs: 400,
        callbacks: tfvis.show.fitCallbacks(
            { name: "训练过程" },
            ["loss"]
        )
    });


    const output = model.predict(tf.tensor([6]));
    output.print();

    console.log(output.dataSync())
}