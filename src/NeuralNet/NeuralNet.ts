import { NEURAL_NET_FILE_NAME, Config, FixedSizeArray, INeuralNet, INode, Layer, NeuralNetMemory, LoggingLevel, MAX_NUMBER_SIZE } from '../types';
import { convertToBinaryArray, generateRandomNumber, generateRandomWeight, getNodeClass, saveFile, shuffle } from '../utils';
import BinaryNode from './nodes/binaryNode';
import PrimaryNode from './nodes/primaryNode';
import initJson from '../../neural_config.json';
import { PrintedNode } from './nodes/node';
import { NNLogger } from './NNLogger';

export class NeuralNet {
    public completedCycles: number;
    public learningRate: number;
    public batchSize: number;
    public totalErrorPerBatch: number[];
    private maximumStoredErrors: number;
    public layers: INeuralNet = [];
    private logger: NNLogger;
    private trainingList: [number, number][] = [];

    public constructor(memory: NeuralNetMemory, loggingLevel: LoggingLevel = LoggingLevel.Default) {
        this.logger = new NNLogger(loggingLevel);
        this.logger.log(LoggingLevel.Default, ['Initializing neural net']);
        const config = initJson as Config;
        this.completedCycles = memory.completedCycles;
        this.learningRate = config.learningRate;
        this.batchSize = config.batchSize;
        this.totalErrorPerBatch = memory.totalErrorPerBatch;
        this.maximumStoredErrors = config.maximumStoredErrors;

        if (!memory.layers.length) {
            const layers = this.createLayers(config.layers);
            memory.layers = layers.map(layer => layer.map(innerLayer => innerLayer.print()));
            this.layers = layers;
            this.saveChanges();
        } else {
            this.loadNeuralNetFromMemory(memory)
        }

        this.logger.log(LoggingLevel.Default, ['Finished initializing neural net']);
    }

    /**
     * Saves the latest changes to the neural net
     */
    public saveChanges = (): void => {
        const newMemory: NeuralNetMemory = {
            completedCycles: this.completedCycles,
            totalErrorPerBatch: this.totalErrorPerBatch,
            layers: this.printLayers()
        }
        saveFile(NEURAL_NET_FILE_NAME, newMemory);
    }

    public saveTotalError = (totalErrorArray: FixedSizeArray<8, number>): void => {
        const totalError = totalErrorArray.reduce((totalError: number, error: number) => totalError + error, 0);

        if (this.totalErrorPerBatch.length > this.maximumStoredErrors) {
            this.totalErrorPerBatch.shift();
        }

        this.totalErrorPerBatch.push(totalError);
    }

    /**
     * Will generate a training list for all the potential number sets that could be passed to the neural net
     */
    private generateTrainingList = () => {
        const newTrainingList: [number, number][] = [];
        for (let i = 0; i <= MAX_NUMBER_SIZE; i++) {
            for (let j = 0; j <= MAX_NUMBER_SIZE; j++) {
                let newTrainingSet: [number, number] = [i, j];
                newTrainingList.push(newTrainingSet);
            }
        }

        const shuffledTrainingList = shuffle<[number, number]>(newTrainingList)
        this.logger.log(LoggingLevel.Inspect, ['New training set:', shuffledTrainingList]);
        this.trainingList = shuffledTrainingList;
    }

    /**
     * Returns the next set of numbers to train on
     * 
     * @returns the next set of training numbers
     */
    private getNextTrainingSet = (): [number, number] => {
        if (this.trainingList.length === 0) {
            this.generateTrainingList();
        }

        return this.trainingList.pop() as [number, number];
    }

    /**
     * Runs a batch training
     */
    public runTrainingBatch = () => {
        const batchSize = this.batchSize;

        let totalErrorArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
        for (let i = 0; i < batchSize; i++) {

            const [firstNumber, secondNumber] = this.getNextTrainingSet();
            const binaryInputArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;

            const product = firstNumber * secondNumber;
            const productArray = convertToBinaryArray(product);

            this.logger.log(LoggingLevel.Inspect, ['Training for product of:', firstNumber, secondNumber]);
            this.logger.log(LoggingLevel.Verbose, ['Binary Array:', binaryInputArray]);
            const resultBinaryArray = this.getProduct(binaryInputArray);
            this.logger.log(LoggingLevel.Verbose, ["Result Array:", resultBinaryArray]);
            this.logger.log(LoggingLevel.Verbose, ["Product Array:", productArray]);

            const errorArray = this.getOutputError(productArray);
            this.logger.log(LoggingLevel.Verbose, ["Error Array:", errorArray]);

            totalErrorArray = totalErrorArray.map((error: number, index: number) => error + errorArray[index]) as FixedSizeArray<8, number>;
        }

        totalErrorArray = totalErrorArray.map(error => error / batchSize) as FixedSizeArray<8, number>;
        this.backPropogate(totalErrorArray);
        this.saveTotalError(totalErrorArray);

        this.completedCycles += 1;
        return totalErrorArray;
    }

    public getProduct = (binaryArray: FixedSizeArray<16, 1 | 0>): FixedSizeArray<8, number> => {
        this.logger.log(LoggingLevel.Verbose, ['Getting product']);
        this.insertInputIntoNeuralNet(binaryArray);
        this.runCalculations();
        const resultBinaryArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
        this.layers[this.layers.length - 1].forEach((node, index) => {
            resultBinaryArray[index] = node.activation;
        });

        this.logger.log(LoggingLevel.Verbose, ['Finished geting product'])
        return resultBinaryArray;
    }

    public getOutputError = (productBinaryArray: FixedSizeArray<8, 1 | 0>): FixedSizeArray<8, number> => {
        this.logger.log(LoggingLevel.Verbose, ['Getting cost']);
        const costArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
        productBinaryArray.forEach((bit: 1 | 0, index: number) => {
            costArray[index] = bit - this.layers[this.layers.length - 1][index].activation;
        })
        return costArray;
    }

    public backPropogate = (totalErrorArray: FixedSizeArray<8, number>) => {
        this.logger.log(LoggingLevel.Verbose, ['Starting backpropogation']);
        let currentErrorArray: number[] = totalErrorArray;
        for (let i = this.layers.length - 1; i > 0; i--) {
            this.logger.log(LoggingLevel.Verbose, ['Current error array:', currentErrorArray, i]);
            const layer = this.layers[i];
            currentErrorArray = this.trainLayer(layer, currentErrorArray);
        }
        this.logger.log(LoggingLevel.Verbose, ['Completed backpropogation']);
    }

    private trainLayer = (layer: INode[], errorArray: number[]): number[] => {
        let newErrorArray = new Array(layer[0].getPrevLayerLength()).fill(1);
        for (let i = 0; i < layer.length; i++) {
            const error = errorArray[i];
            const node = layer[i];
            const nodeErrorArray = node.train(error, this.learningRate);
            newErrorArray = newErrorArray.map((newError: number, index: number) => newError + nodeErrorArray[index]);
        }
        return newErrorArray;
    }

    private runCalculations = () => {
        this.logger.log(LoggingLevel.Verbose, ['Running calculations']);
        const layers = this.layers;
        for (let i = 1; i < layers.length; i++) {
            const layer = layers[i];
            for (let j = 0; j < layer.length; j++) {
                layer[j].calculate();
            }
        }
    }

    private insertInputIntoNeuralNet = (binaryArray: FixedSizeArray<16, 1 | 0>) => {
        this.logger.log(LoggingLevel.Verbose, ['Inserting input into Neural Net']);
        this.layers[0].forEach((node: INode, index: number) => {
            const bit = binaryArray[index];
            node.calculate(bit);
        })
    }

    private createLayers = (layers: Layer[]) => {
        this.logger.log(LoggingLevel.Verbose, ['Creating layers'])
        const generatedLayers: INode[][] = new Array(layers.length);

        for (let i = 0; i < layers.length; i++) {
            const layer = layers[i];
            const prevLayer = i > 0 ? generatedLayers[i - 1] : undefined;
            const nodeClass = getNodeClass(layer.type);
            generatedLayers[i] = this.createLayer(nodeClass, layer.amount, prevLayer);
        }

        this.logger.log(LoggingLevel.Verbose, ['Finished creating layers'])
        return generatedLayers;
    }

    private createLayer = (nodeClass: typeof PrimaryNode | typeof BinaryNode, amount: number, prevLayer?: INode[]) => {
        const newLayer = [];
        for (let i = 0; i < amount; i++) {
            const prevWeights: number[] = [];
            prevLayer?.forEach(_ => {
                prevWeights.push(generateRandomWeight());
            })
            newLayer.push(new nodeClass(i, undefined, prevLayer, prevWeights));
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
        this.logger.log(LoggingLevel.Verbose, ['Loading Neural Net from memory']);
        const layerMemory = data.layers;
        const nodes: INeuralNet = new Array(layerMemory.length).fill([]);

        for (let i = 0; i < layerMemory.length; i++) {
            const layer = layerMemory[i];
            const prevLayer = i > 0 ? nodes[i - 1] : undefined;
            const newLayer: INode[] = [];

            for (let j = 0; j < layer.length; j++) {
                const printedNode = layer[j];
                const nodeClass = getNodeClass(printedNode.type)
                newLayer.push(new nodeClass(j, printedNode.bias, prevLayer, printedNode.prev));
            }

            nodes[i] = newLayer;
        }

        this.layers = nodes;
    };

}