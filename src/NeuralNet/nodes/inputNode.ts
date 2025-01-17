import { INode, NodeType } from "../../types";
import { Bit, Node } from "./node";

export default class InputNode extends Node {
    constructor(index: number, bias?: number, prev?: INode[], prevWeights?: number[]) {
        super(NodeType.Input, index, 0, prev, prevWeights);
    }

    public override calculate(input: number) {
        if (input === undefined) {
            throw 'Did not receive input for the calculation';
        }

        this.activation = input;
    }
}
