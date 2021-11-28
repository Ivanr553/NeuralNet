import data from '../neural_data.json';
import cleanData from '../clean_neural_data.json';
import { Args, NeuralNetMemory, NEURAL_NET_FILE_NAME } from './types';
import { NeuralNet } from './NeuralNet';
import { saveFile } from './utils';


export default (arg: Args) => {
    console.log('Running initialization')

    switch (arg) {
        case Args.Reset: {
            console.log('Reseting neural net data');
            saveFile(NEURAL_NET_FILE_NAME, cleanData);
            process.exit();
        }
        default: ;
    }

    const neuralMemory = data as NeuralNetMemory;
    return new NeuralNet(neuralMemory);
}