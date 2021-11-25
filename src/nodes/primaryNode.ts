import { INode, NodeType } from "../types";
import { Node } from "./node";

export default class PrimaryNode extends Node {
    constructor(index: number, prev?: INode[]) {
        super(NodeType.Primary, index, prev);
    }
}