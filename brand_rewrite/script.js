import { getImages } from "./data";
import * as tfvis from "@tensorflow/tfjs-vis";

window.onload = async () => {
    const { inputs, labels } = await getImages();
    console.log(tfvis);


    const surface = tfvis.visor().surface(
        { name: "IMGS", styles: { height: 200 } }
    )

    inputs.forEach((img) => {
        surface.drawArea.appendChild(img);
    });
};