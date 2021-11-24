import initJson from '../neural_config.json';
import data from '../neural_data.json'
import BinaryNode from './nodes/binaryNode';
import { INode } from './nodes/node';
import PrimaryNode from './nodes/primaryNode';
import { Config, Layer, NeuralNet, NodeType } from './types';
import fs from 'fs';

const NEURAL_NET_FILE_NAME = './neural_data.json';

const createLayers = (layers: Layer[]) => {
    const generatedLayers: INode[][] = new Array(layers.length);

    for(let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        const prevLayer = i > 0 ? generatedLayers[i-1] : undefined;

        switch(layer.type) {
            case NodeType.Binary.toString(): {
                generatedLayers[i] = createLayer(BinaryNode, layer.amount, prevLayer);
                break;
            }

            case NodeType.Primary.toString(): {
                generatedLayers[i] = createLayer(PrimaryNode, layer.amount, prevLayer);
                break;
            }

            default:
                throw `Invalid type of ${layer.type}`;
        }
    }

    return generatedLayers;
}

const createLayer = (nodeClass: typeof PrimaryNode | typeof BinaryNode, amount: number, prevLayer?: INode[]) => {
    const newLayer = [];
    for(let i = 0; i < amount; i++) {
        newLayer.push(new nodeClass(i, prevLayer));
    }
    return newLayer;
}

const saveFile = (fileName: string, json: NeuralNet) => {
    console.log(`Saving file: ${fileName}`);
    fs.writeFileSync(fileName, JSON.stringify(json));
}

export default () => {
    console.log('Running initialization')
    const config = initJson as Config;
    const neuralMemory: NeuralNet = data;

    if (!neuralMemory.layers.length) {
        console.log('Creating new layers');
        const layers = createLayers(config.layers).map(layer => layer.map(innerLayer => innerLayer.print()));
        neuralMemory.layers = layers;
        console.log('Completed creating new layers');
    }

    saveFile(NEURAL_NET_FILE_NAME, neuralMemory);
}