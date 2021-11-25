import { NeuralNet } from '../NeuralNet';
import memory from '../../neural_data.json';
import { FixedSizeArray, NeuralNetMemory } from '../types';
import { convertToBinaryArray } from '../utils';

describe('base tests', () => {
    it('properly updates the entry layer node with the input numbers', () => {
        const NN = new NeuralNet(memory as NeuralNetMemory);
        const firstNumber = 7;
        const secondNumber = 14;
        const binaryArray = [...convertToBinaryArray(firstNumber), ...convertToBinaryArray(secondNumber)] as FixedSizeArray<16, 1 | 0>;
        NN.insertInputIntoNeuralNet(binaryArray);

        binaryArray.forEach((bit: 1 | 0, index: number) => {
            expect(bit).toEqual(NN.nodes[0][index].activation);
        })
    })
})