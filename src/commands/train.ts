import { NeuralNet } from "../NeuralNet/NeuralNet";
import { getCLIResponse } from "../utils";
import cliProgress from 'cli-progress';

export default async (NN: NeuralNet) => {
    console.log('How many cycles?')
    const response = await getCLIResponse('Cycles: ');
    const numberOfCycles = parseInt(response);

    if (numberOfCycles === NaN) {
        throw 'Invalid input given for number of cycles';
    }

    console.log(`Training for ${numberOfCycles} cycles`);
    console.time('Training');

    const progress = new cliProgress.SingleBar({}, cliProgress.Presets.legacy);
    progress.start(numberOfCycles, 0);
    for (let i = 0; i < numberOfCycles; i++) {
        progress.increment();
        NN.runTrainingBatch();
    }
    progress.stop();
    
    console.timeEnd('Training');
}