import init from './init';
import { Args } from './types';

const arg = process.argv.slice(2)[0];

init(arg as Args);
