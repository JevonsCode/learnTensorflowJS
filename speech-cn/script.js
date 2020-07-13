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
import * as speechCommands from "@tensorflow-models/speech-commands"

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