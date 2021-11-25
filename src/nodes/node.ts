import { INode, NodeType } from "../types";
import { sigmoid } from "../utils";

export interface PrevNode {
    index: number;
    node: INode,
    weight: number;
}

export interface PrintedNode {
    type: NodeType;
    bias: number;
    prev: number[];
}

/**
 * Base abstract class for nodes. Every type of node will extend this.
 */
export abstract class Node implements INode {
    public type: NodeType;
    public index: number;
    private bias: number = 0;
    private prev: PrevNode[] | undefined;
    public activation: number = 0;

    public constructor(type: NodeType, index: number, prev?: INode[]) {
        this.type = type;
        this.index = index;
        this.bias = 0;
        this.prev = [];
        if (prev) {
            this.prev = prev.map(node => ({
                index: node.index,
                node,
                weight: 0
            }));
        }
    }

    /**
     * Calculates the activation of the node
     * 
     * @param input The binary number input used in binary nodes
     */
    public calculate(input?: 1 | 0): void {
        if (!this.prev) {
            throw 'Attempting to calculate on a node without a list of previous nodes';
        }
        const prevActivations = this.prev.slice().reduce<number>((accumulatedValue: number, prevNode: PrevNode): number => accumulatedValue + (prevNode.weight * prevNode.node.activation), 0);
        this.activation = sigmoid(prevActivations + this.bias);
    };

    /**
     * Resets the neuron for the next calculation
     */
    public reset(): void {
        this.activation = 0;
    }

    /**
     * Trains the node to learn from the testing result
     */
    public train(): void {

    }

    /**
     * Returns a formatted object that can be used to store the node in the json memory file
     * 
     * @returns a printed node object
     */
    public print(): PrintedNode {
        const printedNode: PrintedNode = {
            type: this.type,
            bias: this.bias,
            prev: [],
        };

        if (this.prev) {
            printedNode.prev = this.prev.slice().map(node => node.weight);
            return printedNode;
        } else {
            return printedNode;
        }
    }
}