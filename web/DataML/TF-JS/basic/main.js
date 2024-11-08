async function predict(){ 

document.getElementById('output_field').innerText = "Running 1";
//import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('./model.json');

const xp = tf.tensor2d([[83.28,9.94,0.37,0.74,5.95,1.43,100.28,-15.9,0.36,-21.4,0.36,-40.3,8.4,48.6,47.4,2.15,1.19,24.6,1.99,0.073,0.242,0.296,0.054,17.57,5.57,18.43,28.03,14.44,3.63,11.86,0.46,15.96,7.83,22.54,0.64,0.05,10.58,18.56,16.85,51.97,1.99,8.48,1.78,10.9,2.5,13.4,17.1,24.7,32.7,18,7.4,12.7,24.4,38.1,18.4,6.5,791,4739,4739,5.99]]);

document.getElementById('output_field').innerText = xp;

const prediction = model.predict(xp);

document.getElementById('output_field').innerText = prediction.dataSync();
}

predict();
