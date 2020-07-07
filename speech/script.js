/**
 * 语音识别
 */

import * as speechCommands from "@tensorflow-models/speech-commands";

const MODEL_PATH = "http://127.0.0.1:8080/speech";

window.onload = async () => {
    const recognizer = speechCommands.create(
        // 浏览器自带的傅里叶变换（FFT：傅里叶变换）
        "BROWSER_FFT",
        null,
        MODEL_PATH + "/model.json",
        MODEL_PATH + "/metadata.json"
    );

    await recognizer.ensureModelLoaded();

    const words = recognizer.wordLabels();

    words.forEach(word => {
        const node = document.createElement("div");
        node.innerText = word;
        document.getElementById("speech").appendChild(node);
    });

    console.log(words);

    recognizer.listen(result => {
        const { scores } = result;
        const whichone = Math.max(...scores);
        const _index = scores.indexOf(whichone);
        console.log(words[_index]);

        words.forEach((word, index) => {
            const w = document.getElementById("speech").getElementsByTagName("div")[index];

            if (_index === index) {
                w.style = "background: #77e9"
            } else {
                w.style = "background: #fff"
            }
        });
    }, {
        includeSpectrogram: true,
        // 概率阈值 The callback function will be invoked if and only if the maximum probability score of all the words is greater than this threshold. Default: 0.
        probabilityThreshold: 0.9,
        // 频率 Controls how often the recognizer performs prediction on spectrograms. Must be >=0 and <1 (default: 0.5).
        // For example, if each spectrogram is 1000 ms long and overlapFactor is set to 0.25, 
        // the prediction will happen every 250 ms.
        overlapFactor: 0.3
    })
}