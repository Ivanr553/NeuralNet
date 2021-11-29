import { LoggingLevel } from "../types";

export class NNLogger {
    private loggingLevel: LoggingLevel;

    constructor(loggingLevel: LoggingLevel) {
        this.loggingLevel = loggingLevel;
    }

    public log(loggingLevel: LoggingLevel, log: (string | number | boolean | [] | object)[]) {
        if (this.loggingLevel >= loggingLevel) {
            console.log(...log);
        }
    }
}