import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { IRIS_CLASSES, IRIS_NUM_CLASSES, getIrisData } from "./data";

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: "sigmoid"
    }));

    model.add(tf.layers.dense({
        // 神经元个数应该等于输出类别个数
        units: 3,
        activation: "softmax"
    }));

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.1),
        metrics: ["accuracy"]
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        // 验证集
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            { name: "训练结果" },
            ["loss", "val_loss", "acc", "val_acc"],
            { callbacks: ["onEpochEnd"] }
        )
    });









    document.getElementById("btn").disabled = false;

    window.predict = () => {
        const a = document.getElementById("a");
        const b = document.getElementById("b");
        const c = document.getElementById("c");
        const d = document.getElementById("d");
        const pred = model.predict(tf.tensor([[a.value * 1, b.value * 1, c.value * 1, d.value * 1]]));
        console.log(pred, IRIS_CLASSES[pred.argMax(1).dataSync(0)])
    }
}