import { NeuralNet } from "../NeuralNet/NeuralNet";

export default async (NN: NeuralNet, number?: number) => {
    const {expectedNumber, guess, guessNumberArray, expectedResultNumberArray, activations} = NN.guessMnistNumber(number);

    console.log('Correct answer:', expectedNumber);
    console.log('Guess:', guess);
    console.log('Expected Number Array', expectedResultNumberArray);
    console.log('Guess guessNumberArray:', guessNumberArray);
	console.log('Activations:', activations)
}
