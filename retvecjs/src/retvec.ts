// @ts-ignore
import * as tf from '@tensorflow/tfjs/dist/tf.fesm.js';

export default class RetVec {
    static encodingSize:number = 24;

    /* this is built in the init */
    static masks:number[] = [];

    // loaded model
    static model: tf.GraphModel;

    // where to load the tokenizer model from
    static modelPath:string;

    // are we using the tokenizer model?
    static use_model:boolean;


    static async init(modelPath = "../node_modules/@google-research/retvecjs/model/v1/model.json", encodingSize = 24) {
        let p:number = 1;
        for(let i = 24; i; i--) {
            RetVec.masks.push(p);
            p = p * 2;
        }
        RetVec.masks.reverse();
        RetVec.encodingSize = encodingSize;
        RetVec.modelPath = modelPath;
        if (modelPath != '') {
            await RetVec._loadModel();
            RetVec.use_model = true;
        } else {
            RetVec.use_model = false;
        }
    }

    private static async _loadModel() {
        RetVec.model = await tf.loadGraphModel(RetVec.modelPath);
    }

    static binarizer(input: string | string[], strLen:number=16): tf.Tensor | Array<tf.Tensor> {
        if (Array.isArray(input)) {
            const result: Array<tf.Tensor> = input.map(word => {
                let vect:Array<number[]> = new Array();
                let vlen:number = Math.min(strLen, word.length);

                // agnostic binary coversion
                for (let i = 0; i < vlen; i++) {

                    let charbin:number[] = []
                    let val:number = word.charCodeAt(i);
                    for (let midx = 0; midx < RetVec.encodingSize; midx++) {
                        let v:number = val & RetVec.masks[midx] ? 1 : 0;
                        charbin.push(v);
                    }
                    vect.push(charbin);
                }

                const wpad:Array<number> = new Array(RetVec.encodingSize).fill(0);
                for (let i = vlen; i < strLen; i++) {
                    vect.push(wpad)
                }
                return tf.tensor(vect);
            })
            return result
        } else {
            let vect:Array<number[]> = new Array();
            let vlen:number = Math.min(strLen, input.length);

            // agnostic binary coversion
            for (let i = 0; i < vlen; i++) {

                let charbin:number[] = []
                let val:number = input.charCodeAt(i);
                for (let midx = 0; midx < RetVec.encodingSize; midx++) {
                    let v:number = val & RetVec.masks[midx] ? 1 : 0;
                    charbin.push(v);
                }
                vect.push(charbin);
            }

            const wpad:Array<number> = new Array(RetVec.encodingSize).fill(0);
            for (let i = vlen; i < strLen; i++) {
                vect.push(wpad)
            }
            return tf.tensor(vect);
        }
    }

    static embed(input: tf.Tensor | Array<tf.Tensor>, batchSize = 32): Array<tf.Tensor> | tf.Tensor {
        if (Array.isArray(input)) {
            const result:Array<tf.Tensor> = input.map(enc => {
                const inputs: tf.Tensor = tf.expandDims(enc as tf.Tensor, 0);
                const result = RetVec.model.predict(inputs, {batchSize: batchSize}) as tf.Tensor
                return tf.expandDims(result.dataSync(), 0)
            })
            return result
        } else {
            const inputs: tf.Tensor = tf.expandDims(input as tf.Tensor, 0);
            const result = RetVec.model.predict(inputs, {batchSize: batchSize}) as tf.Tensor
            return tf.expandDims(result.dataSync(), 0)
        }
    }

    static tokenizer(input: string | string[], strLen:number=16, batchSize=32): Array<tf.Tensor> | tf.Tensor {
        const binEncodings = RetVec.binarizer(input, strLen)

        if (RetVec.use_model) {
            const embeddings = RetVec.embed(binEncodings, batchSize)
            return embeddings;
        } else {
            return binEncodings;
        }
    }
}