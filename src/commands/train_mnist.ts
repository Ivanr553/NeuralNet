import { NeuralNet } from '../NeuralNet/NeuralNet';
import { LoggingLevel, ModelType } from '../types';
import { getCLIResponse } from '../utils';
import cliProgress from 'cli-progress';

export default async (NN: NeuralNet, logginglevel: LoggingLevel, number?: number) => {
	console.log("Running Mnist training");

	try {
		console.log('How many cycles?')
		const response = await getCLIResponse('Cycles: ');
		const numberOfCycles = parseInt(response);

		if (numberOfCycles === NaN) {
			throw 'Invalid input given for number of cycles';
		}

		console.log(`Training for ${numberOfCycles} cycles`);
		console.time('Training');

		if(number !== undefined) {
			NN.useOnlyOneNumberMnist(number);
		}

		if(logginglevel === LoggingLevel.Verbose) {
			for (let i = 0; i < numberOfCycles; i++) {
				NN.runTrainingBatch(ModelType.Mnist);
			}
		} else {
			const progress = new cliProgress.SingleBar({}, cliProgress.Presets.legacy);
			progress.start(numberOfCycles, 0);
			for (let i = 0; i < numberOfCycles; i++) {
				progress.increment();
				NN.runTrainingBatch(ModelType.Mnist);
			}
			progress.stop();
		}

		console.timeEnd('Training');	
	} catch (error) {
		console.error(error);	
	}
}
