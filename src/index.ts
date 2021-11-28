import init from './init';
import runTest from './runTest';
import { Args, FixedSizeArray } from './types';
import { generateRandomNumber, getCLIResponse } from './utils';

(async () => {
    const arg = process.argv.slice(2)[0];

    const NN = init(arg as Args);

    const response = await getCLIResponse('How many cycles?');
    const numberOfCycles = parseInt(response);

    if (numberOfCycles === NaN) {
        throw 'Invalid input given for number of cycles';
    }

    let totalErrorArray: FixedSizeArray<8, number> = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < numberOfCycles; i++) {
        const firstNumber = generateRandomNumber(0, 10);
        const secondNumber = generateRandomNumber(0, 10);

        const errorArray = runTest(NN, firstNumber, secondNumber);
        totalErrorArray = totalErrorArray.map((error: number, index: number) => error + errorArray[index]) as FixedSizeArray<8, number>;
    }

    totalErrorArray = totalErrorArray.map(error => error / numberOfCycles) as FixedSizeArray<8, number>;

    const totalError = totalErrorArray.reduce((totalError: number, error: number) => totalError + error, 0);
    console.log("Total Error:", totalError);

    NN.backPropogate(totalErrorArray);
})()

