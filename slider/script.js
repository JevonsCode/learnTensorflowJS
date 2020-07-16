import * as speechCommands from "@tensorflow-models/speech-commands";


const MODEL_PATH = "http://127.0.0.1:8080/speech";

const speechRecognizer;

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

    await recognizer.ensureModelLoaded();
}