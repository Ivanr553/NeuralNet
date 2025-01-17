import { NEURAL_NET_FILE_NAME, Config, FixedSizeArray, INeuralNet, INode, Layer, NeuralNetMemory, LoggingLevel, MAX_NUMBER_SIZE, NEURAL_NET_COST_FILE_NAME, ModelType, NodeType } from '../types';
import { convertToBinaryArray, generateRandomNumber, generateRandomWeight, getNodeClass, ReLuDerivative, saveFile, shuffle } from '../utils';
import PrimaryNode from './nodes/primaryNode';
import initJson from '../../neural_config.json';
import { Bit, PrintedNode } from './nodes/node';
import { NNLogger } from './NNLogger';
import totalErrorJson from '../../totalError.json';
import InputNode from './nodes/inputNode';
import OutputNode from './nodes/outputNode';
import { loadMnist } from '../mnist';

export class NeuralNet {
	public completedCycles: number;
	public learningRate: number;
	public batchSize: number;
	private maximumStoredErrors: number;
	public layers: INeuralNet = [];
	private logger: NNLogger;
	private trainingList: [number, number][] = [];

	//Mnist Training Data
	private images: number[][] = [];
	private labels: number[] = [];
	private mnistIndexes: number[] = [];
	private isUsingOneNumber: boolean = false;
	private oneNumberIndex: number = 0;

	public constructor(memory: NeuralNetMemory, loggingLevel: LoggingLevel = LoggingLevel.Default, modelType: ModelType) {
		this.logger = new NNLogger(loggingLevel);
		this.logger.log(LoggingLevel.Default, ['Initializing neural net']);
		const config = initJson as Config;
		this.completedCycles = memory.completedCycles;
		this.learningRate = config.learningRate;
		this.batchSize = config.batchSize;
		this.maximumStoredErrors = config.maximumStoredErrors;
		this.loadMnistTrainingData();

		if (!memory.layers.length) {
			const layers = this.createLayers(modelType, config.layers);
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
			layers: this.printLayers()
		}
		saveFile(NEURAL_NET_FILE_NAME, newMemory);
	}

	public saveTotalError = (totalErrorArray: number[]): void => {
		const totalError = totalErrorArray.reduce((totalError: number, error: number) => totalError + Math.abs(error), 0);

		const totalErrorJsonParsed = totalErrorJson as {cost: number[]};

		if (totalErrorJsonParsed.cost.length > this.maximumStoredErrors) {
			totalErrorJsonParsed.cost.shift();
		}

		totalErrorJsonParsed.cost.push(totalError);

		saveFile(NEURAL_NET_COST_FILE_NAME, totalErrorJsonParsed);
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

	public useOnlyOneNumberMnist = (number: number) => {
		const newImages: number[][] = [];
		const newLabels: number[] = [];

		this.labels.forEach((label, index) => {
			if(label === number) {
				newImages.push(this.images[index]);
				newLabels.push(number);
			}	
		});

		this.images = newImages;
		this.labels = newLabels;

		this.isUsingOneNumber = true;
	}

	/**
		* Loads the mnist training data into the class
	*/
	private loadMnistTrainingData = () => {
		const {images, labels} = loadMnist();

		this.images = images;
		this.labels = labels;

		const initialIndexArray = [];
		for(let i = 0; i < this.images.length; i++) {
			initialIndexArray.push(i);
		}

		this.mnistIndexes = shuffle<number>(initialIndexArray);		
	}

	private getNextMnistIndex = (): number => {
		if(this.isUsingOneNumber) {
			if(this.oneNumberIndex === this.images.length) {
				this.oneNumberIndex = 0;	
			}
			return this.oneNumberIndex++;
		}

		if(!this.mnistIndexes.length) {
			const initialIndexArray = [];
			for(let i = 0; i < this.images.length; i++) {
				initialIndexArray.push(i);
			}

			this.mnistIndexes = shuffle<number>(initialIndexArray);		
		}	

		return this.mnistIndexes.pop() as number;
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
	public runTrainingBatch = (trainingType: ModelType, number?: number) => {
		switch(trainingType) {
			case ModelType.Product: {
				return this.runProductTrainingBatch(number);
			}

			case ModelType.Mnist: {
				return this.runMnistTrainingBatch();
			}

			default:
				throw `No trainingType of ${trainingType}`;
		}
	}

	private runMnistTrainingBatch = () => {
		const batchSize = this.batchSize;

		let totalErrorArray: FixedSizeArray<10, number> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		for(let i = 0; i < this.batchSize; i++) {
			const index = this.getNextMnistIndex();
			const image = this.images[index].map(num => num/255);
			const label = this.labels[index];

			this.logger.log(LoggingLevel.Verbose, ['Expected Number:', label]);
			this.insertInputIntoNeuralNet(image);

			const resultArray = this.runCalculations();
			const expectedResultNumberArray = this.getExpectedNumberResultArray(label);
			const errorArray = this.getOutputError(resultArray, expectedResultNumberArray);

			this.resetLayerActivations();
			totalErrorArray = totalErrorArray.map((error: number, index: number) => error + errorArray[index]) as FixedSizeArray<10, number>;
		}

		totalErrorArray = totalErrorArray.map(error => error / batchSize) as FixedSizeArray<10, number>;
		// console.log(totalErrorArray);
		this.backPropogate(totalErrorArray);
		this.saveTotalError(totalErrorArray);

		this.completedCycles += 1;
		return totalErrorArray;
	}

	private getExpectedNumberResultArray = (resultNumber: number) => {
		const resultNumberArray: number[] = [];
		for(let i = 0; i < 10; i++) {
			resultNumberArray.push(i === resultNumber ? 1 : 0);
		}
		return resultNumberArray;
	}

	private runProductTrainingBatch = (number?: number) => {
		const batchSize = this.batchSize;

		let totalErrorArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
		for (let i = 0; i < batchSize; i++) {
			const [firstNumber, secondNumber] = number ? [number, number] : this.getNextTrainingSet();
			const binaryInputArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, Bit>;
			const product = firstNumber * secondNumber;
			const expectedProduct = convertToBinaryArray(product);

			this.logger.log(LoggingLevel.Inspect, ['Training for product of:', firstNumber, secondNumber]);
			this.logger.log(LoggingLevel.Verbose, ['Binary Array:', binaryInputArray]);
			const resultBinaryArray = this.getProduct(binaryInputArray);
			this.logger.log(LoggingLevel.Verbose, ["Result Array:", resultBinaryArray]);
			this.logger.log(LoggingLevel.Verbose, ["Expected Product Array:", expectedProduct]);

			const errorArray = this.getOutputError(resultBinaryArray, expectedProduct);
			this.logger.log(LoggingLevel.Verbose, ["Error Array:", errorArray]);

			totalErrorArray = totalErrorArray.map((error: number, index: number) => error + errorArray[index]) as FixedSizeArray<8, number>;
		}

		totalErrorArray = totalErrorArray.map(error => error / batchSize) as FixedSizeArray<8, number>;
		console.log(totalErrorArray);
		this.backPropogate(totalErrorArray);
		this.saveTotalError(totalErrorArray);
		this.resetLayerActivations();

		this.completedCycles += 1;
		return totalErrorArray;
	}

	private getNumber = (image: number[]): {activations: number[], guess: number} => {
		this.logger.log(LoggingLevel.Verbose, ['Getting number']);
		this.insertInputIntoNeuralNet(image);
		const activations = this.runCalculations();

		const resultNumberObj: {
			number: number | undefined,
			activation: number
		} = {
			number: undefined,
			activation: -Infinity
		};
		this.layers[this.layers.length - 1].forEach(({activation}, number) => {
			if(resultNumberObj.number === undefined || resultNumberObj.activation < activation) {
				resultNumberObj.number = number;
				resultNumberObj.activation = activation;
			}

		}, {number: undefined, activation: -Infinity});

		if(resultNumberObj.number === undefined) {
			throw 'Number not generated from neural net';
		}

		this.logger.log(LoggingLevel.Verbose, ['Finished geting product']);
		return {activations, guess: resultNumberObj.number};
	}

	public guessMnistNumber = (number?: number): {
		expectedNumber: number,
		guess: number,
		guessNumberArray: number[],
		expectedResultNumberArray: number[],
		activations: number[]
	} => {
		number !== undefined && this.useOnlyOneNumberMnist(number);

		const randomIndex = Math.floor(generateRandomNumber(this.images.length - 1, 0));
		console.log('index', randomIndex);
		const image = this.images[randomIndex];
		console.log('image length:', image.length);
		const expectedNumber = this.labels[randomIndex];
		const expectedResultNumberArray = this.getExpectedNumberResultArray(expectedNumber);

		const {activations, guess} = this.getNumber(image);
		const guessNumberArray = this.getExpectedNumberResultArray(guess);

		return {expectedNumber, guess, guessNumberArray, expectedResultNumberArray, activations};
	}

	public getProduct = (binaryArray: FixedSizeArray<16, Bit>): FixedSizeArray<8, number> => {
		this.logger.log(LoggingLevel.Verbose, ['Getting product']);
		this.insertInputIntoNeuralNet(binaryArray);
		this.runCalculations();
		const resultBinaryArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
		this.layers[this.layers.length - 1].forEach((node, index) => {
			resultBinaryArray[index] = node.activation;
		});

		this.logger.log(LoggingLevel.Verbose, ['Finished geting product']);
		return resultBinaryArray;
	}

	private resetLayerActivations() {
		this.logger.log(LoggingLevel.Verbose, ['Resetting layers']);
		const layers = this.layers;
		for (let i = 1; i < layers.length; i++) {
			const layer = layers[i];
			for (let j = 0; j < layer.length; j++) {
				layer[j].reset();
			}
		}
	}

	public getOutputError = (resultBinaryArray: number[], expectedResultNumberArray: number[]): number[] => {
		this.logger.log(LoggingLevel.Verbose, ['Getting cost']);
		const costArray: number[] = new Array(resultBinaryArray.length).fill(0);
		expectedResultNumberArray.forEach((bit: number, i: number) => {
			const cost = bit - ReLuDerivative(resultBinaryArray[i]);
			// console.log(resultBinaryArray[i], bit, cost);
			costArray[i] = cost;
		})
		return costArray;
	}

	public backPropogate = (totalErrorArray: number[]) => {
		this.logger.log(LoggingLevel.Verbose, ['Starting backpropogation']);
		this.logger.log(LoggingLevel.Verbose, ['totalErrorArray', totalErrorArray]);
		let currentErrorArray: number[] = totalErrorArray;
		for (let i = this.layers.length - 1; i > 0; i--) {
			const layer = this.layers[i];
			currentErrorArray = this.trainLayer(layer, currentErrorArray);
		}
		this.logger.log(LoggingLevel.Verbose, ['Completed backpropogation']);
	}

	private trainLayer = (layer: INode[], errorArray: number[]): number[] => {
		const prevLayerLength = layer[0].getPrevLayerLength();

		let newErrorArray = new Array(prevLayerLength).fill(1);
		for (let i = 0; i < layer.length; i++) {
			const error = errorArray[i];
			const node = layer[i];
			this.logger.log(LoggingLevel.Verbose, ['Node error', error]);
			// this.logger.log(LoggingLevel.Verbose, ['Node starting values', node.print()]);
			const nodeErrorArray = node.train(error, this.learningRate);
			// this.logger.log(LoggingLevel.Verbose, ['Node ending values', node.print()]);
			newErrorArray = newErrorArray.map((newError: number, index: number) => newError + nodeErrorArray[index]);
		}


		newErrorArray = newErrorArray.map(error => error / prevLayerLength) as FixedSizeArray<8, number>;
		return newErrorArray;
	}

	private runCalculations = (): number[] => {
		this.logger.log(LoggingLevel.Verbose, ['Running calculations']);
		const resultArray = [];

		const layers = this.layers;
		for (let i = 1; i < layers.length; i++) {
			const layer = layers[i];
			for (let j = 0; j < layer.length; j++) {
				layer[j].calculate();
				if(i === layers.length - 1) {
					resultArray.push(layer[j].activation);
				}
			}
		}

		return resultArray;
	}

	private insertInputIntoNeuralNet = (binaryArray: number[]) => {
		this.logger.log(LoggingLevel.Verbose, ['Inserting input into Neural Net']);
		this.layers[0].forEach((node: INode, index: number) => {
			const input = binaryArray[index];
			node.calculate(input);
		})
	}

	private createLayers = (modelType: ModelType, layers: Layer[]) => {
		this.logger.log(LoggingLevel.Verbose, ['Creating layers'])

		// Adding 2 to account for input and output layers
		const generatedLayers: INode[][] = new Array(layers.length + 2);

	let firstLayerLength = 0;

	switch(modelType) {
		case ModelType.Product: {
			firstLayerLength = 16;
			break
		}

		case ModelType.Mnist: {
			firstLayerLength = 28 * 28;
			break;
		}

		default:
			throw `Invalid modelType of ${modelType} when generating layers`;
	}

	const firstLayerNodeClass = getNodeClass(NodeType.Input);
	generatedLayers[0] = this.createLayer(firstLayerNodeClass, firstLayerLength, undefined);

	for (let i = 0; i < layers.length; i++) {
		const layer = layers[i];
		const prevLayer = generatedLayers[i];
		const nodeClass = getNodeClass(layer.type);
		generatedLayers[i + 1] = this.createLayer(nodeClass, layer.amount, prevLayer);
	}

	let lastLayerLength = 0; 
	switch(modelType) {
		case ModelType.Product: {
			lastLayerLength = 8;
			break
		}

		case ModelType.Mnist: {
			lastLayerLength = 10;
			break;
		}

		default:
			throw `Invalid modelType of ${modelType} when generating layers`;
	}
	const lastLayerNodeClass = getNodeClass(NodeType.Output);
	generatedLayers[generatedLayers.length - 1] = this.createLayer(lastLayerNodeClass, lastLayerLength, generatedLayers[generatedLayers.length - 2]);

	this.logger.log(LoggingLevel.Verbose, ['Finished creating layers'])
	return generatedLayers;
	}

	private createLayer = (nodeClass: typeof PrimaryNode | typeof InputNode | typeof OutputNode, amount: number, prevLayer?: INode[]) => {
		const newLayer = [];
		for (let i = 0; i < amount; i++) {
			const prevWeights: number[] = [];
			prevLayer?.forEach(_ => {
				prevWeights.push(generateRandomWeight());
			})
			const bias = generateRandomNumber(1, -1);
			newLayer.push(new nodeClass(i, bias, prevLayer, prevWeights));
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
