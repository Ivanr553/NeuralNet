import { PrintedNode } from "./nodes/node";

export const NEURAL_NET_FILE_NAME = './neural_data.json';

export enum Args {
    Reset = 'reset'
}

export interface INode {
    index: number;
    activation: number;
    calculate(input?: 1 | 0): void;
    print: () => PrintedNode;
}

export enum NodeType {
    Primary = 'Primary',
    Binary = 'Binary'
}

export interface Config {
    layers: Layer[];
}

export interface Layer {
    type: NodeType;
    amount: number;
}

export type INeuralNet = INode[][];

export interface NeuralNetMemory {
    trainingCycles: number;
    failures: number;
    layers: PrintedNode[][];
}

export type FixedSizeArray<N extends number, T> = N extends 0 ? never[] : {
    0: T;
    length: N;
} & Array<T>;