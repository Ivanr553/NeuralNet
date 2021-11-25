import { FixedSizeArray, INode } from "./types";
import { convertToBinaryArray } from "./utils";
import { NeuralNet } from "./NeuralNet";



export const runTest = (NN: NeuralNet, firstNumber: number, secondNumber: number) => {
    const answer = firstNumber * secondNumber;
    const binaryArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;

    NN.insertInputIntoNeuralNet(binaryArray);
}