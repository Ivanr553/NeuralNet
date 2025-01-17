import config from '../neural_config.json';
import { Bit, PrintedNode } from "./NeuralNet/nodes/node";

export const NEURAL_NET_FILE_NAME = './neural_data.json';
export const NEURAL_NET_COST_FILE_NAME = './totalError.json';

export const MAX_NUMBER_SIZE = 15;

export enum Args {
    Reset = 'reset',
    Train = 'train',
    Guess = 'guess',
	TrainMnist = 'train-m',
	GuessMnist = 'guess-m'
}

export enum LoggingLevel {
    Default = 0,
    Inspect = 1,
    Verbose = 2
}

export interface INode {
    index: number;
    activation: number;
	ReLUActivation: number;
    calculate(input?: Bit | number): void;
    print: () => PrintedNode;
    train: (error: number, learningRate: number) => number[];
    getPrevLayerLength: () => number;
	reset: () => void;
}

export enum NodeType {
    Primary = 'Primary',
	Output = 'Output',
	Input = 'Input'
}

export interface Config {
    batchSize: number;
    maximumStoredErrors: number;
    learningRate: number;
    layers: Layer[];
}

export interface Layer {
    type: NodeType;
    amount: number;
}

export type INeuralNet = INode[][];

export interface NeuralNetMemory {
    completedCycles: number;
    layers: PrintedNode[][];
}

export type FixedSizeArray<N extends number, T> = N extends 0 ? never[] : {
    0: T;
    length: N;
} & Array<T>;

export enum ModelType {
	Product,
	Mnist
}
