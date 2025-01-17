import cleanData from '../clean_neural_data.json';
import data from '../neural_data.json';
import { Args, ModelType, NeuralNetMemory, NEURAL_NET_COST_FILE_NAME, NEURAL_NET_FILE_NAME } from './types';
import { saveFile } from './utils';
import argv from 'minimist';
import { NeuralNet } from './NeuralNet/NeuralNet';
import train from './commands/train';
import guess from './commands/guess';
import train_mnist from './commands/train_mnist';
import guessMnist from './commands/guessMnist';

(async () => {
	try {
		const args = argv(process.argv.slice(2));

		const shouldReset = args._.indexOf(Args.Reset) > -1;
		if (shouldReset) {
			console.log('Resetting neural net data');
			saveFile(NEURAL_NET_FILE_NAME, cleanData);
			saveFile(NEURAL_NET_COST_FILE_NAME, {cost: []})
			process.exit();
		}

		const loggingLevel = args?.level;
		loggingLevel !== undefined && console.log('Logging level:', loggingLevel);

		const number: number | undefined = args?.number;
		number !== undefined && console.log('training only against:', number);

		const neuralMemory = data as any as NeuralNetMemory;

		if (args._.indexOf(Args.Train) > -1) {
			const NN = new NeuralNet(neuralMemory, loggingLevel, ModelType.Product);
			await train(NN, loggingLevel, number);
			NN.saveChanges();
		} else if (args._.indexOf(Args.Guess) > -1) {
			const NN = new NeuralNet(neuralMemory, loggingLevel, ModelType.Product);
			guess(NN);
		} else if (args._.indexOf(Args.TrainMnist) > -1) {
			const NN = new NeuralNet(neuralMemory, loggingLevel, ModelType.Mnist);
			train_mnist(NN, loggingLevel, number);
		} else if (args._.indexOf(Args.GuessMnist) > -1) {
			const NN = new NeuralNet(neuralMemory, loggingLevel, ModelType.Mnist);
			guessMnist(NN, number);
		}  else {
			console.log('No process argument given. Choose from these options:');
			console.log(' - ', Args.Train);
			console.log(' - ', Args.TrainMnist);
			console.log(' - ', Args.Guess);
			console.log(' - ', Args.Reset);
		}
	} catch (error) {
		console.error(error)	
	}
})()
