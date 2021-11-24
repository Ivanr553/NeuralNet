import { PrintedNode } from "./nodes/node";

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

export interface NeuralNet {
    trainingCycles: number;
    failures: number;
    layers: PrintedNode[][]
}