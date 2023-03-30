class Chooser {
    static add_number(numbers: NumberSet): void {
        let length = numbers.length;
        let n = 2;
        while (numbers.length == length) {
            for (let m = 1; m < n && numbers.length == length; ++m) {
                numbers.add(m/n);
            }
        }
    }
    static choose_weight(): number {
        return 0;
    }
    static choose_element<T>(elements: Array<T>): T {
        return elements[Chooser.choose_index(elements.length)];
    }
    static *choose_order<T>(elements: Array<T>): Generator<T,void,undefined> {
        while (elements.length > 0) {
            let index = Chooser.choose_index(elements.length);
            yield* elements.splice(index, 1);
        }
    }
    static choose_face(faces: Array<Polygon>): Polygon {
        return Chooser.choose_element(faces);
    }
    static choose_index(length: number): number {
        return 0;
    }
}

class RandomChooser extends Chooser {
    static add_number(numbers: NumberSet) {
        numbers.add( Math.random() * (numbers.max() - numbers.min())
            + numbers.min() );
    }
    static choose_weight(): number {
        return Math.random();
    }
    static choose_index(length: number) {
        let index = Math.floor(length * Math.random());
        if (index >= length)
            return index - 1;
        return index;
    }
}

