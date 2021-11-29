import { INode, NodeType } from "../../types";
import { activationDerivative, sigmoid } from "../../utils";

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

    public constructor(type: NodeType, index: number, bias?: number, prev?: INode[], prevWeights?: number[]) {
        this.type = type;
        this.index = index;
        this.bias = bias || 0;
        this.prev = [];
        if (prev && prevWeights) {
            this.prev = prev.map((node, index) => ({
                index: node.index,
                node,
                weight: prevWeights[index]
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
     * Gets the length of the prev layer
     * 
     * @returns the length of the prev layer
     */
    public getPrevLayerLength = (): number => {
        return this.prev ? this.prev.length : 0;
    }

    /**
     * Trains the node to learn from the testing result
     * 
     * @param error the error used to recalculate this node's weights
     * @returns the error array for this node's connections
     */
    public train(error: number, learningRate: number): number[] {
        if (!this.prev) {
            return [];
        }

        // adjust bias
        const biasDelta = learningRate * error * activationDerivative(this.activation);
        const currentBias = this.bias;
        this.bias = currentBias + biasDelta;

        const newError: number[] = [];
        this.prev.forEach(prevNode => {
            // adjust weight
            const delta = learningRate * error * activationDerivative(this.activation) * prevNode.node.activation;
            const currentWeight = prevNode.weight;
            prevNode.weight = currentWeight + delta;

            // calculate error for next layer
            const errorProportion = prevNode.weight * error;
            newError.push(errorProportion);
        })

        return newError;
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