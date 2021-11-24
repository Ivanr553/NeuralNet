
export interface INode {
    index: number;
    calculate(): void;
    print: () => PrintedNode;
}

export interface PrevNode {
    index: number;
    node: INode,
    weight: number;
}

export interface PrintedNode {
    bias: number;
    prev: number[];
}

export abstract class Node implements INode {
    public index: number;
    private bias: number = 0;
    private prev: PrevNode[] | undefined;
    public activation: number = 0;

    public constructor(index: number, prev?: INode[]) {
        this.index = index;
        if(prev) {
            this.prev = prev.map(node => ({
                index: node.index,
                node,
                weight: 0
            }));
        }
    }

    abstract calculate(): void;

    public print() {
        const printedNode: PrintedNode = {
            bias: this.bias,
            prev: [],
        };

        if(this.prev) {
            printedNode.prev = this.prev.slice().map(node => node.weight);
            return printedNode;
        } else {
            return printedNode;
        }
    }
}