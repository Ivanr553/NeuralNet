import config from '../neural_config.json';
import BinaryNode from './nodes/binaryNode';
import PrimaryNode from './nodes/primaryNode';
import { FixedSizeArray, NeuralNetMemory, NodeType } from './types';
import fs from 'fs';

export const sigmoid = (z: number): number => {
    return 1 / (1 + Math.exp(-z / config.sigmoidConstant));
}

export const getNodeClass = (nodeType: NodeType): typeof BinaryNode | typeof PrimaryNode => {
    switch (nodeType) {
        case NodeType.Binary.toString(): {
            return BinaryNode;
        }

        case NodeType.Primary.toString(): {
            return PrimaryNode;
        }

        default:
            throw `Unable to get node class. Invalid type of ${nodeType}`;
    }
}

export const convertToBinaryArray = (number: number): FixedSizeArray<8, 1 | 0> => {
    const binaryArray: FixedSizeArray<8, 1 | 0> = [0, 0, 0, 0, 0, 0, 0, 0];

    if (number < 0) {
        throw 'Attempting to convert negative number to binary array';
    }

    if (number > 255) {
        throw `${number} is too large to be converted to a byte array`;
    }

    let count = 0;
    let numberCopy = number;
    while (numberCopy / 2 > 0) {
        let newNumber = numberCopy / 2;
        if (newNumber % 1 !== 0) {
            binaryArray[binaryArray.length - 1 - count] = 1;
            newNumber = Math.floor(newNumber);
        }
        numberCopy = newNumber;
        count++;
    }

    return binaryArray;
}

export const saveFile = (fileName: string, json: NeuralNetMemory) => {
    console.log(`Saving file: ${fileName}`);
    fs.writeFileSync(fileName, JSON.stringify(json));
}

//numbers need to be smaller than 16