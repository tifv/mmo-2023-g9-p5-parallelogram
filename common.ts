const EPSILON = 1e-7;

function epsilon_sign(value: number): number {
    return (value > EPSILON ? 1 : 0) - (value < -EPSILON ? 1 : 0);
}

function modulo(a: number, n: number): number {
    return ((a % n) + n) % n;
}

function get_or_die<K,V>(map: Map<K,V>, key: K): V {
    let value = map.get(key);
    if (value === undefined)
        throw new Error("The key does not belong to the map")
    return value;
}

function* itermap<A,B>(values: Iterable<A>, mapper: (value: A) => B):
    IterableIterator<B>
{
    for (let value of values)
        yield mapper(value);
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
    find_key(key: number, exact?: true):
        {index: number, key: number, value: V}
    find_key(key: number, exact: false):
        {index: number, key: undefined, value: undefined} |
        {index: number, key: number, value: V}
    find_key(key: number, exact: boolean = true):
        {index: number, key?: number, value?: V}
    {
        let
            min_index = 0, max_index = this.length,
            cursor_index = min_index;
        while(max_index > min_index) {
            cursor_index = Math.floor((min_index + max_index) / 2);
            let [cursor_key, cursor_value] = this[cursor_index];
            if (cursor_key > key + EPSILON) {
                max_index = cursor_index;
            } else if (cursor_key < key - EPSILON) {
                min_index = cursor_index + 1;
            } else {
                return { index: cursor_index,
                    key: cursor_key, value: cursor_value };
            }
        }
        if (exact)
            throw new Error("Number is not in the map.");
        return {index: min_index};
    }
    set(key: number, value: V): [number, V] {
        let item = this.find_key(key, false);
        if (item.key !== undefined) {
            return [item.key, item.value];
        }
        this.splice(item.index, 0, [key, value]);
        return [key, value];
    }
    get(key: number): V {
        return this.find_key(key).value;
    }
    find_value(value: V): {index: number, key: number} {
        for (let i = 0; i < this.length; ++i) {
            let [cursor_key, cursor_value] = this[i];
            if (cursor_value === value)
                return {index: i, key: cursor_key};
        }
        throw new Error("Value is not in the map.");
    }
}

class SlicedArray<V> extends NumberMap<V[]> {
    add_guard(number: number) {
        this.push([number, []]);
    }
    slice_values(start_key?: number, end_key?: number): Array<V> {
        let start_index: number = (start_key !== undefined) ?
            this.find_key(start_key).index : 0;
        let sliced_values = new Array<V>();
        for (let i = start_index; i < this.length; ++i) {
            let [cursor_key, values] = this[i];
            if (end_key !== undefined) {
                if (cursor_key == end_key)
                    break;
                if (cursor_key > end_key)
                    throw new Error("End number is not in the map.");
            }
            sliced_values.push(...values);
        }
        return sliced_values;
    }
}

type LPModel = {
    optimize: any,
    opType: any,
    variables: {[x :string]: {[x: string]: number}},
    constraints: { [x: string]:
        {max?: number, min?: number, equal?: number}
    },
};

function lpsolve(model: LPModel): { [s: string]: any; } {
    // @ts-ignore
    return solver.Solve(model);
}

