namespace Choosers {

export class Chooser {
    offspring(): Chooser {
        return this;
    }
    reset(): void {
    }
    random(): number {
        return 0.5;
    }
    coin(probability: number): boolean {
        return this.random() > probability;
    }
    add_number(numbers: NumberSet): void {
        let ranges = new Array<[number, number]>();
        {
            let prev_number = null;
            for (let number of numbers) {
                if (prev_number === null) {
                    prev_number = number;
                    continue;
                }
                ranges.push([prev_number, number]);
            }
        }
        if (ranges.length < 1)
            throw new Error( "Cannot add an intermediate number " +
                "in a set with less than 2 numbers" );
        let [total_weight, weights] = ranges.reduce(
            ([total, weights], [x, y]) => {
                 let weight = (y - x)**2;
                 weights.push(weight);
                 return [total + weight, weights];
            }, [0, new Array<number>()] );
        let chooser = this.offspring();
        let chosen_weight = chooser.random() * total_weight;
        let [[x, y]] = ranges.filter( (range, index) =>
            weights[index] > chosen_weight );
        numbers.add(chooser.random() * (y - x) + x);
    }
    choose_weight(): number {
        return 2 * this.random() - 1;
    }
    choose_element<T>(elements: Array<T>): T {
        return elements[this.choose_index(elements.length)];
    }
    *choose_order<T>(elements: Array<T>): Iterable<T> {
        let chooser = this.offspring()
        while (elements.length > 0) {
            let index = chooser.choose_index(elements.length);
            yield* elements.splice(index, 1);
        }
    }
    choose_face(faces: Array<Polygon>): Polygon {
        return this.choose_element(faces);
    }
    choose_index(length: number) {
        let index = Math.floor(length * this.random());
        if (index >= length)
            return index - 1;
        return index;
    }
}

export class RandomChooser extends Chooser {
    random(): number {
        return Math.random();
    }
    add_number(numbers: NumberSet) {
        numbers.add( this.random() * (numbers.max() - numbers.min())
            + numbers.min() );
    }
}

function gcd(x: number, y: number): number {
    x = Math.abs(Math.round(x));
    y = Math.abs(Math.round(y));
    while (y > 0) {
        if (x < y) {
            [x, y] = [y, x];
            continue;
        }
        [x, y] = [y, x % y];
    }
    return x;
}

const GOLD = (-1 + Math.sqrt(5)) / 2;

export class CoprimeChooser extends Chooser {
    modulo: number;
    shift: number;
    value: number;

    constructor(
        modulo: number,
        shift: number,
        initial: number,
    ) {
        super();
        this.modulo = modulo;
        this.shift = shift;
        this.value = initial;
    }
    static from_modulo(modulo: number): CoprimeChooser {
        let shift = CoprimeChooser._generate_shift(modulo);
        return new CoprimeChooser(modulo, shift, 0);
    }
    static default() {
        return CoprimeChooser.from_modulo(2*3*5*7*11);
    }
    private static _generate_shift(modulo: number): number {
        let shift_candidates = [];
        for (let i = 1; i < modulo; ++i) {
            shift_candidates.push(i);
        }
        shift_candidates.sort( (i, j) =>
            (Math.abs(i - GOLD*modulo) - Math.abs(j - GOLD*modulo)) );
        for (let i of shift_candidates) {
            if (gcd(i, modulo) > 1)
                continue;
            return i;
        }
        // not really reachable, but whatever
        return 1;
    }
    offspring(): CoprimeChooser {
        let value = this.value;
        this.random();
        return new CoprimeChooser(
            this.modulo, this.shift, value );
    }
    reset(): void {
        this.value = 0;
    }
    random(): number{
        let result = this.value / this.modulo;
        this.value = (this.value + this.shift) % this.modulo;
        return result;
    }
}

}

import Chooser = Choosers.Chooser;