import { NEURAL_NET_FILE_NAME, Config, FixedSizeArray, INeuralNet, INode, Layer, NeuralNetMemory } from './types';
import { getNodeClass, saveFile } from './utils';
import BinaryNode from './nodes/binaryNode';
import PrimaryNode from './nodes/primaryNode';
import initJson from '../neural_config.json';

export class NeuralNet {
    public nodes: INeuralNet = [];

    public constructor(memory: NeuralNetMemory) {
        console.log('Initializing neural net');
        const config = initJson as Config;

        if (!memory.layers.length) {
            console.log('Creating new layers');
            const layers = this.createLayers(config.layers);
            const printedLayers = layers.map(layer => layer.map(innerLayer => innerLayer.print()));
            memory.layers = printedLayers;
            this.nodes = layers;
            console.log('Completed creating new layers');
            saveFile(NEURAL_NET_FILE_NAME, memory);
        } else {
            console.log('Loaded from memory');
            this.loadNeuralNetFromMemory(memory)
        }

        console.log('Finished initializing neural net');
    }

    public getNodes = () => {
        return this.nodes;
    }

    public insertInputIntoNeuralNet = (binaryArray: FixedSizeArray<16, 1 | 0>) => {
        this.nodes[0].forEach((node: INode, index: number) => {
            const bit = binaryArray[index];
            node.calculate(bit);
        })
    }

    private createLayers = (layers: Layer[]) => {
        const generatedLayers: INode[][] = new Array(layers.length);

        for (let i = 0; i < layers.length; i++) {
            const layer = layers[i];
            const prevLayer = i > 0 ? generatedLayers[i - 1] : undefined;
            const nodeClass = getNodeClass(layer.type);
            generatedLayers[i] = this.createLayer(nodeClass, layer.amount, prevLayer);
        }

        return generatedLayers;
    }

    private createLayer = (nodeClass: typeof PrimaryNode | typeof BinaryNode, amount: number, prevLayer?: INode[]) => {
        const newLayer = [];
        for (let i = 0; i < amount; i++) {
            newLayer.push(new nodeClass(i, prevLayer));
        }
        return newLayer;
    }

    private loadNeuralNetFromMemory = (data: NeuralNetMemory): void => {
        const layerMemory = data.layers;
        const nodes: INeuralNet = new Array(layerMemory.length).fill([]);

        for (let i = 0; i < layerMemory.length; i++) {
            const layer = layerMemory[i];
            const prevLayer = i > 0 ? nodes[i - 1] : undefined;
            const newLayer: INode[] = [];

            for (let j = 0; j < layer.length; j++) {
                const printedNode = layer[j];
                const nodeClass = getNodeClass(printedNode.type)
                newLayer.push(new nodeClass(printedNode.bias, prevLayer));
            }

            nodes[i] = newLayer;
        }

        this.nodes = nodes;
    };

}