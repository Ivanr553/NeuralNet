import cleanData from '../clean_neural_data.json';
import data from '../neural_data.json';
import { Args, NeuralNetMemory, NEURAL_NET_FILE_NAME } from './types';
import { saveFile } from './utils';
import argv from 'minimist';
import { NeuralNet } from './NeuralNet/NeuralNet';
import train from './commands/train';
import guess from './commands/guess';

(async () => {
    const args = argv(process.argv.slice(2));

    const shouldReset = args._.indexOf(Args.Reset) > -1;
    if (shouldReset) {
        console.log('Resetting neural net data');
        saveFile(NEURAL_NET_FILE_NAME, cleanData);
        process.exit();
    }

    const loggingLevel = args?.level;
    loggingLevel !== undefined ? console.log('Logging level:', loggingLevel) : null;

    const neuralMemory = data as any as NeuralNetMemory;
    const NN = new NeuralNet(neuralMemory, loggingLevel);

    if (args._.indexOf(Args.Train) > -1) {
        await train(NN);
        NN.saveChanges();
    } else if (args._.indexOf(Args.Guess) > -1) {
        guess(NN);
    } else {
        console.log('No process argument given. Choose from these options:');
        console.log(' - ', Args.Train);
        console.log(' - ', Args.Guess);
        console.log(' - ', Args.Reset);
    }
})()
