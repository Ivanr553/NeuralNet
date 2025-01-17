import { INode, NodeType } from "../../types";
import { Node } from "./node";

export default class OutputNode extends Node {
    constructor(index: number, bias?: number, prev?: INode[], prevWeights?: number[]) {
        super(NodeType.Output, index, 0, prev, prevWeights);
    }

    public override train(error: number, learningRate: number): number[] {
        if (!this.prev) {
            return [];
        }

        const newError: number[] = [];
        this.prev.forEach(prevNode => {
            // adjust weight
            const delta = learningRate * error; 
            const currentWeight = prevNode.weight;
            prevNode.weight = currentWeight + (currentWeight * delta);

            // calculate error for next layer
            const errorProportion = prevNode.weight * error;
            newError.push(errorProportion);
        })

        return newError;
    }
}
