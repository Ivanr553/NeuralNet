import * as fs from 'fs';
import * as path from 'path';

export const loadMnist = (): {
	images: number[][],
	labels: number[]
} => {
	const imagesPath = path.join(__dirname, '../../mnist', 'train-images.idx3-ubyte');
	const labelsPath = path.join(__dirname, '../../mnist', 'train-labels.idx1-ubyte');

	const imagesBuffer = fs.readFileSync(imagesPath);
	const labelsBuffer = fs.readFileSync(labelsPath);

	const readInt32 = (buffer: Buffer, offset: number): number => buffer.readUInt32BE(offset);

	const readLabels = (buffer: Buffer): number[] => {
		const labels = [];

		for (let i = 0; i < buffer.length; i++) {
			labels.push(buffer.readUInt8(i));
		}

		return labels;
	};

	const readImages = (buffer: Buffer, count: number, rows: number, cols: number): number[][][] => {
		const images = [];

		for (let i = 0; i < count; i++) {
			const image: number[][] = [];

			for (let j = 0; j < rows; j++) {
				const row: number[] = [];

				for (let k = 0; k < cols; k++) {
					const pixel = buffer.readUInt8(i * rows * cols + j * cols + k);
					row.push(pixel);
				}

				image.push(row);
			}

			images.push(image);
		}

		return images;
	};

	const labelsMagic = readInt32(labelsBuffer, 0);
	const labelsCount = readInt32(labelsBuffer, 4);
	const labels = readLabels(labelsBuffer.slice(8));

	const imagesMagic = readInt32(imagesBuffer, 0);
	const imagesCount = readInt32(imagesBuffer, 4);
	const imagesRows = readInt32(imagesBuffer, 8);
	const imagesCols = readInt32(imagesBuffer, 12);
	const imagesData = readImages(imagesBuffer.slice(16), imagesCount, imagesRows, imagesCols);
	const images: number[][] = [];

	for (const imageData of imagesData) {
		const image: number[] = [];

		for (const row of imageData) {
			image.push(...row);
		}

		images.push(image);
	}

	return {
		images,
		labels
	}
}
