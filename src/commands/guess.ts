import { NeuralNet } from "../NeuralNet/NeuralNet";
import { FixedSizeArray } from "../types";
import { convertToBinaryArray, convertToBit, convertToIntFromBinaryArray, getCLIResponse } from "../utils";

export default async (NN: NeuralNet) => {
    console.log('Pick two numbers between 0 and 15');

    const firstNumber = parseInt(await getCLIResponse('First: '));
    if (firstNumber > 15 || firstNumber < 0 || firstNumber === NaN) {
        throw 'Invalid input given for firstNumber';
    }

    const secondNumber = parseInt(await getCLIResponse('Second: '));
    if (secondNumber > 15 || secondNumber < 0 || secondNumber === NaN) {
        throw 'Invalid input given for secondNumber';
    }

    const binaryInputArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;
    const resultBinaryArray = NN.getProduct(binaryInputArray);
    const bitifiedResultBinaryArray = resultBinaryArray.slice().map(activation => convertToBit(activation));
    const guess = convertToIntFromBinaryArray(bitifiedResultBinaryArray);
    const productBinaryArray = convertToBinaryArray(firstNumber * secondNumber);

    console.log('Correct answer:', firstNumber * secondNumber);
    console.log('Guess:', guess);
    console.log('Correct binary array', productBinaryArray);
    console.log('Guess binary array:', resultBinaryArray);
}