import { NeuralNet } from '../NeuralNet/NeuralNet';
import memory from '../../neural_data.json';
import { FixedSizeArray, NeuralNetMemory } from '../types';
import { convertToBinaryArray } from '../utils';

describe('base tests', () => {
    it('properly updates the entry layer node with the input numbers', () => {
        const NN = new NeuralNet(memory as any as NeuralNetMemory);
        const firstNumber = 7;
        const secondNumber = 14;
        const binaryArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;
        NN.getProduct(binaryArray);

        binaryArray.forEach((bit: 1 | 0, index: number) => {
            expect(bit).toEqual(NN.layers[0][index].activation);
        })
    })
})
