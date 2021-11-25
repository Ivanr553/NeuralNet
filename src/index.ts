import init from './init';
import runTest from './runTest';
import { Args } from './types';
import { generateRandomNumber } from './utils';

const arg = process.argv.slice(2)[0];

const NN = init(arg as Args);

const firstNumber = generateRandomNumber(0, 10);
const secondNumber = generateRandomNumber(0, 10);

runTest(NN, firstNumber, secondNumber);
