import { FixedSizeArray } from "./types";
import { convertToBinaryArray } from "./utils";
import { NeuralNet } from "./NeuralNet";

export default (NN: NeuralNet, firstNumber: number, secondNumber: number): FixedSizeArray<8, number> => {
    console.time('Run');

    const product = firstNumber * secondNumber;
    const productArray = convertToBinaryArray(product);
    const binaryArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;

    console.log('Getting product of:', firstNumber, secondNumber);
    console.log('Binary Array:', binaryArray);
    const resultBinaryArray = NN.getProduct(binaryArray);
    console.log("Result:", resultBinaryArray);
    console.log("Product Array:", productArray);

    const errorArray = NN.getError(productArray)
    console.log("Error Array:", errorArray);

    console.timeEnd('Run');
    return errorArray;
}