/**
 * 语音识别
 *
 * 迁移学习
 * 用其他模型预处理
 * 启 hs data 服务
 *
 *
 *
 */
import * as speechCommands from "@tensorflow-models/speech-commands";
import * as tfvis from "@tensorflow/tfjs-vis";

const MODEL_PATH = "http://127.0.0.1:8080/speech";

let transferRecognizer;

window.onload = async () => {
    document.querySelector("i").innerHTML = "模型准备中...";

    // 识别器
    const recognizer = speechCommands.create(
        "BROWSER_FFT",
        // 词汇
        null,
        // 自定义模型的 url
        MODEL_PATH + "/model.json",
        MODEL_PATH + "/metadata.json",
    );

    // 确保加载完成
    await recognizer.ensureModelLoaded();

    transferRecognizer = recognizer.createTransfer("我是一串字符串");

    document.querySelector("i").innerHTML = "模型准备完成!";
}



window.collect = async (btn) => {
    btn.disabled = true;

    const label = btn.innerText;
    await transferRecognizer.collectExample(
        label === "背景噪音" ? "_background_noise_" : label
    );
    btn.disabled = false;

    document.querySelector("#count").innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 4);

    console.log(transferRecognizer.countExamples())
}

window.train = async () => {
    await transferRecognizer.train({
        epochs: 30,
        callback: tfvis.show.fitCallbacks(
            { name: "train" },
            ["loss", "acc"],
            { callbacks: ["onEpochEnd"] }
        )
    })
}

window.toggle = async (checked) => {
    if (checked) {
        await transferRecognizer.listen(result => {
            const { scores } = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            const r = labels[index];
            console.log(`这是个啥 ==> : window.toggle -> r`, r)
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.85
        });
    } else {
        transferRecognizer.stopListening();
    }
}


/**
 * @link https://github.com/tensorflow/tfjs-models/tree/master/speech-commands#serialize-examples-from-a-transfer-recognizer
 */
window.save = () => {
    const arrayBuffer = transferRecognizer.serializeExamples();
    const b = new Blob([arrayBuffer]);
    const a = document.createElement("a");
    a.href = window.URL.createObjectURL(b);
    a.download = "data.bin";
    a.click();
}