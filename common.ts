const EPSILON = 1e-7;

function epsilon_sign(value: number): number {
    return (value > EPSILON ? 1 : 0) - (value < -EPSILON ? 1 : 0);
}

function get_or_die<K,V>(map: Map<K,V>, key: K): V {
    let value = map.get(key);
    if (value === undefined)
        throw new Error("The key does not belong to the map")
    return value;
}

class NumberSet extends Array<number> {
    static from_numbers(numbers: Iterable<number>): NumberSet {
        let set = new NumberSet();
        for (let number of numbers) {
            set.add(number);
        }
        return set;
    }
    add(number: number): number {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let current = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return current;
            }
        }
        this.splice(min_index, 0, number);
        return number;
    }
    min(): number {
        return this[0];
    }
    max(): number {
        return this[this.length - 1];
    }
}

class NumberMap<V> extends Array<[number,V]> {
    set(number: number, value: V): [number, V] {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return [current, current_value];
            }
        }
        this.splice(min_index, 0, [number, value]);
        return [number, value];
    }
    get(number: number): V {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return current_value;
            }
        }
        throw new Error("Number is not in the map.");
    }
    index_of_key(number: number): number {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return index;
            }
        }
        throw new Error("Number is not in the map.");
    }
    index_of_value(value: V): number {
        for (let i = 0; i < this.length; ++i) {
            let [, current_value] = this[i];
            if (current_value === value)
                return i;
        }
        throw new Error("Value is not in the map.");
    }
}

class SlicedArray<V> extends NumberMap<V[]> {
    add_guard(number: number) {
        this.push([number, []]);
    }
    slice_values(start?: number, end?: number): Array<V> {
        let start_index: number = start !== undefined ? 
            this.index_of_key(start) : 0;
        let sliced_values = new Array<V>();
        for (let i = start_index; i < this.length; ++i) {
            let [current, values] = this[i];
            if (end !== undefined) {
                if (current == end)
                    break;
                if (current > end)
                    throw new Error("End number is not in the map.");
            }
            sliced_values.push(...values);
        }
        return sliced_values;
    }
}

