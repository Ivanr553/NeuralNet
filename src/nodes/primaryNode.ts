import { Node, INode } from "./node";

export default class PrimaryNode extends Node implements INode {
    constructor(index: number, prev?: INode[]) {
        super(index, prev);
    }

    calculate() {
        
    }
}