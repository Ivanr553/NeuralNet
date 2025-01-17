import PrimaryNode from './NeuralNet/nodes/primaryNode';
import { FixedSizeArray, NodeType } from './types';
import fs from 'fs';
import readline from 'readline';
import OutputNode from './NeuralNet/nodes/outputNode';
import { Bit } from './NeuralNet/nodes/node';
import InputNode from './NeuralNet/nodes/inputNode';

export const ReLU = (z: number): number => {
	if (z > 0) {
		return z;
	} else {
		return 0;
	}
}

export const activationDerivative = (activation: number): number => {
    return activation * (1 - activation);
}

export const ReLuDerivative = (activation: number): number => {
	if (activation < 0) return 0;
	return 1;
}

export const generateRandomNumber = (max: number, min: number) => Math.random() * (max - min + 1) + min;
export const generateRandomWeight = () => {
    const isNegative = Math.random() > 0.5;
    if (isNegative) {
        return -(generateRandomNumber(1, 0));
    } else {
        return generateRandomNumber(1, 0);
    }
}

export const getNodeClass = (nodeType: NodeType): typeof InputNode | typeof PrimaryNode | typeof OutputNode => {
    switch (nodeType) {
        case NodeType.Input.toString(): {
            return InputNode;
        }

        case NodeType.Primary.toString(): {
            return PrimaryNode;
        }

		case NodeType.Output.toString(): {
			return OutputNode;
		}

        default:
            throw `Unable to get node class. Invalid type of ${nodeType}`;
    }
}

export const convertIntArrayToBinaryArray = (array: FixedSizeArray<8, number>): FixedSizeArray<8, Bit> => {
	return array.slice().map(num => convertToBit(num)) as FixedSizeArray<8, Bit>;
}

export const convertToBinaryArray = (number: number): FixedSizeArray<8, Bit> => {
    const binaryArray: FixedSizeArray<8, Bit> = [0, 0, 0, 0, 0, 0, 0, 0];

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

export const convertToBit = (num: number): Bit => {
    return num >= 1 ? 1 : 0;
}

export const convertToIntFromBinaryArray = (binaryArray: Bit[]): number => {
    let number = 0;
    for (let i = binaryArray.length - 1; i >= 0; i--) {
        const power = (binaryArray.length - 1) - i;
        let currentBinaryDecimal = Math.pow(2, power);
        number += binaryArray[i] * currentBinaryDecimal;
    }
    return number;
}

export const saveFile = (fileName: string, json: Object) => {
    fs.writeFileSync(fileName, JSON.stringify(json));
}

export const getCLIResponse = (question: string): Promise<string> => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise(resolve => rl.question(question, (ans: string) => {
        rl.close();
        resolve(ans);
    }))
}

export const shuffle = <T>(array: T[]): T[] => {
    let currentIndex = array.length, randomIndex;

    while (currentIndex != 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }

    return array;
}


//numbers need to be smaller than 16
