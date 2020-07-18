import * as speechCommands from "@tensorflow-models/speech-commands";
import { div } from "@tensorflow/tfjs";


const MODEL_PATH = "http://127.0.0.1:8080/speech";

let transFerRecognizer;
let currentIndex = 0;

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

    transFerRecognizer = recognizer.createTransfer("slider");

    const res = await fetch(MODEL_PATH + "/../slider/data.bin");

    const arrayBuffer = await res.arrayBuffer();

    transFerRecognizer.loadExamples(arrayBuffer);

    // console.log("==> ", transFerRecognizer.countExamples());

    await transFerRecognizer.train({ epochs: 60 });

    console.log("done!")

    document.querySelector("i").innerHTML = "模型准备完成！";
}

window.toggle = async (checked) => {
    if (checked) {
        await transFerRecognizer.listen(result => {
            const { scores } = result;
            console.log(`这是个啥 ==> : window.toggle -> scores`, scores)
            const labels = transFerRecognizer.wordLabels();
            console.log(`这是个啥 ==> : window.toggle -> labels`, labels)
            const index = scores.indexOf(Math.max(...scores));
            console.log(labels[index]);
            window.sliderPlay(labels[index]);
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.6
        });
    } else {
        transFerRecognizer.stopListening();
    }
}

window.sliderPlay = (label) => {
    const imgBox = document.querySelector(".img-box");
    console.log(`这是个啥 ==> : window.sliderPlay -> imgBox`, imgBox);
    if (label === "上一张") {
        if (currentIndex !== 0)
            currentIndex -= 1;
    } else {
        if (currentIndex !== document.querySelectorAll(".img").length - 1)
            currentIndex += 1;
    }
    imgBox.style.transform = `translateX(-${500 * currentIndex}px)`
}
