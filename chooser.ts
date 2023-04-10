class TrivialChooser {
    add_number(numbers: NumberSet): void {
        let length = numbers.length;
        let n = 2;
        while (numbers.length == length) {
            for (let m = 1; m < n && numbers.length == length; ++m) {
                numbers.add(m/n);
            }
        }
    }
    choose_weight(): number {
        return 0;
    }
    choose_element<T>(elements: Array<T>): T {
        return elements[this.choose_index(elements.length)];
    }
    *choose_order<T>(elements: Array<T>): Generator<T,void,undefined> {
        while (elements.length > 0) {
            let index = this.choose_index(elements.length);
            yield* elements.splice(index, 1);
        }
    }
    choose_face(faces: Array<Polygon>): Polygon {
        return this.choose_element(faces);
    }
    choose_index(length: number): number {
        return 0;
    }
}

class RandomChooser extends TrivialChooser {
    add_number(numbers: NumberSet) {
        numbers.add( Math.random() * (numbers.max() - numbers.min())
            + numbers.min() );
    }
    choose_weight(): number {
        return 2 * Math.random() - 1;
    }
    choose_index(length: number) {
        let index = Math.floor(length * Math.random());
        if (index >= length)
            return index - 1;
        return index;
    }
}

type Chooser = TrivialChooser;
var Chooser = RandomChooser;
