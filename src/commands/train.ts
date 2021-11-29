import { NeuralNet } from "../NeuralNet/NeuralNet";
import { getCLIResponse } from "../utils";

export default async (NN: NeuralNet) => {
    console.log('How many cycles?')
    const response = await getCLIResponse('Cycles: ');
    const numberOfCycles = parseInt(response);

    if (numberOfCycles === NaN) {
        throw 'Invalid input given for number of cycles';
    }

    console.log(`Training for ${numberOfCycles} cycles`);
    console.time('Training')
    for (let i = 0; i < numberOfCycles; i++) {
        NN.runTrainingBatch();
    }
    console.timeEnd('Training');
}