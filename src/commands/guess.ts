import { NeuralNet } from "../NeuralNet/NeuralNet";
import { FixedSizeArray } from "../types";
import { convertToBinaryArray, convertToBit, convertToIntFromBinaryArray, getCLIResponse } from "../utils";

export default async (NN: NeuralNet) => {
    console.log('Pick two numbers between 0 and 15');

    const firstNumber = parseInt(await getCLIResponse('First: '));
    if (firstNumber > 15 || firstNumber < 0 || firstNumber === NaN) {
        console.log('Invalid input given for firstNumber');
        process.exit();
    }

    const secondNumber = parseInt(await getCLIResponse('Second: '));
    if (secondNumber > 15 || secondNumber < 0 || secondNumber === NaN) {
        console.log('Invalid input given for secondNumber');
        process.exit();
    }

    const binaryInputArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;
    const resultBinaryArray = NN.getProduct(binaryInputArray);
    const bitifiedResultBinaryArray = resultBinaryArray.slice().map(activation => convertToBit(activation));
    const guess = convertToIntFromBinaryArray(bitifiedResultBinaryArray);
    const productBinaryArray = convertToBinaryArray(firstNumber * secondNumber);
    const averageCost = resultBinaryArray.slice().reduce((prev: number, curr: number, index: number) => prev + (Math.pow(curr - productBinaryArray[index], 2)), 0);

    console.log('Correct answer:', firstNumber * secondNumber);
    console.log('Guess:', guess);
    console.log('Correct binary array', productBinaryArray);
    console.log('Guess binary array:', bitifiedResultBinaryArray);
    console.log('Guess output array:', resultBinaryArray);
    console.log('Average cost:', averageCost);
}