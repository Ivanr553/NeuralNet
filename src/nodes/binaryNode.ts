import { INode, NodeType } from "../types";
import { Node } from "./node";

/**
 * Class used for input and output neurons. Input neurons will use acceptInput to set the activation value. Output neurons will use the calculate to determine the final binary value
 */
export default class BinaryNode extends Node {
    constructor(index: number, prev?: INode[]) {
        super(NodeType.Binary, index, prev);
    }

    public override calculate(input: 1 | 0) {
        if (input === undefined) {
            throw 'Did not receive input for BinaryNode calculation';
        }

        this.activation = input;
    }
}