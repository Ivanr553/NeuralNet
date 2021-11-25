import { FixedSizeArray } from "./types";
import { convertToBinaryArray } from "./utils";
import { NeuralNet } from "./NeuralNet";

export default (NN: NeuralNet, firstNumber: number, secondNumber: number) => {
    const product = firstNumber * secondNumber;
    const productArray = convertToBinaryArray(product);
    const binaryArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;

    const resultBinaryArray = NN.getProduct(binaryArray);
    console.log(resultBinaryArray);
}