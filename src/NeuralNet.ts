import { NEURAL_NET_FILE_NAME, Config, FixedSizeArray, INeuralNet, INode, Layer, NeuralNetMemory } from './types';
import { getNodeClass, saveFile } from './utils';
import BinaryNode from './nodes/binaryNode';
import PrimaryNode from './nodes/primaryNode';
import initJson from '../neural_config.json';
import { PrintedNode } from './nodes/node';

export class NeuralNet {
    public trainingCycles: number;
    public batchSize: number;
    public failures: number;
    public layers: INeuralNet = [];

    public constructor(memory: NeuralNetMemory) {
        console.log('Initializing neural net');
        const config = initJson as Config;
        this.trainingCycles = memory.trainingCycles;
        this.batchSize = memory.batchSize;
        this.failures = memory.failures;

        if (!memory.layers.length) {
            const layers = this.createLayers(config.layers);
            memory.layers = layers.map(layer => layer.map(innerLayer => innerLayer.print()));
            this.layers = layers;
            this.saveChanges();
        } else {
            this.loadNeuralNetFromMemory(memory)
        }

        console.log('Finished initializing neural net');
    }

    /**
     * Saves the latest changes to the neural net
     */
    private saveChanges = () => {
        const newMemory: NeuralNetMemory = {
            trainingCycles: this.trainingCycles,
            failures: this.failures,
            batchSize: this.batchSize,
            layers: this.printLayers()
        }
        saveFile(NEURAL_NET_FILE_NAME, newMemory);
    }

    public getProduct = (binaryArray: FixedSizeArray<16, 1 | 0>): FixedSizeArray<8, number> => {
        console.log('Getting product');
        this.insertInputIntoNeuralNet(binaryArray);
        this.runCalculations();
        const resultBinaryNode: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
        this.layers[this.layers.length - 1].forEach((node, index) => {
            resultBinaryNode[index] = node.activation;
        });

        console.log('Finished geting product')
        return resultBinaryNode;
    }

    public getError = (productBinaryArray: FixedSizeArray<8, 1 | 0>): FixedSizeArray<8, number> => {
        console.log('Getting cost');
        const costArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
        productBinaryArray.forEach((bit: 1 | 0, index: number) => {
            costArray[index] = this.layers[this.layers.length - 1][index].activation - bit;
        })
        return costArray;
    }

    public backPropogate = (errorArray: FixedSizeArray<8, number>) => {
        console.log('Starting backpropogation');
        let currentErrorArray: number[] = errorArray;
        for (let i = this.layers.length - 1; i > 0; i--) {
            console.log('Current error array:', currentErrorArray, i);
            const layer = this.layers[i];
            currentErrorArray = this.trainLayer(layer, currentErrorArray);
        }
        this.saveChanges();
        console.log('Completed backpropogation');
    }

    private trainLayer = (layer: INode[], errorArray: number[]): number[] => {
        let newErrorArray = new Array(layer[0].getPrevLayerLength()).fill(1);
        for (let i = 0; i < layer.length; i++) {
            const error = errorArray[i];
            const node = layer[i];
            const nodeErrorArray = node.train(error);
            newErrorArray = newErrorArray.map((newError: number, index: number) => newError + nodeErrorArray[index]);
        }
        return newErrorArray;
    }

    private runCalculations = () => {
        console.log('Running calculations');
        const layers = this.layers;
        for (let i = 1; i < layers.length; i++) {
            const layer = layers[i];
            for (let j = 0; j < layer.length; j++) {
                layer[j].calculate();
            }
        }
    }

    private insertInputIntoNeuralNet = (binaryArray: FixedSizeArray<16, 1 | 0>) => {
        console.log('Inserting input into Neural Net');
        this.layers[0].forEach((node: INode, index: number) => {
            const bit = binaryArray[index];
            node.calculate(bit);
        })
    }

    private createLayers = (layers: Layer[]) => {
        console.log('Creating layers')
        const generatedLayers: INode[][] = new Array(layers.length);

        for (let i = 0; i < layers.length; i++) {
            const layer = layers[i];
            const prevLayer = i > 0 ? generatedLayers[i - 1] : undefined;
            const nodeClass = getNodeClass(layer.type);
            generatedLayers[i] = this.createLayer(nodeClass, layer.amount, prevLayer);
        }

        console.log('Finished creating layers')
        return generatedLayers;
    }

    private createLayer = (nodeClass: typeof PrimaryNode | typeof BinaryNode, amount: number, prevLayer?: INode[]) => {
        const newLayer = [];
        for (let i = 0; i < amount; i++) {
            newLayer.push(new nodeClass(i, prevLayer));
        }
        return newLayer;
    }

    /**
     * Prints the layers and returns them
     * 
     * @returns printed layers
     */
    private printLayers = (): PrintedNode[][] => {
        const printedLayers: PrintedNode[][] = [];
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const printedLayer: PrintedNode[] = [];
            layer.forEach(node => {
                printedLayer.push(node.print());
            })
            printedLayers.push(printedLayer);
        }
        return printedLayers;
    }

    private loadNeuralNetFromMemory = (data: NeuralNetMemory): void => {
        console.log('Loading Neural Net from memory');
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

        this.layers = nodes;
    };

}