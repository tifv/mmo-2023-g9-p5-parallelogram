namespace Algebra {

export namespace Sparse {

export class DimensionError extends Error {};

class NumberArray {
    size: number;
    indices: Array<number>;
    values: Array<number>;

    constructor(
        size: number,
        indices: Array<number>,
        values: Array<number>,
    ) {
        this.size = size;
        this.indices = indices;
        this.values = values;
    }

    as_dense(): Array<number> {
        let dense = new Array<number>(this.size);
        dense.fill(0);
        for (let [index, value] of this.iter_items()) {
            dense[index] = value;
        }
        return dense;
    }

    _copy_as<TArray extends NumberArray>(
        ArrayClass: (new (
            size: number,
            indices: Array<number>,
            values: Array<number>,
        ) => TArray),
    ): TArray {
        return new ArrayClass( this.size,
            Array.from(this.indices), Array.from(this.values) );
    }

    /** @param items [index, value] pairs */
    static _from_items_as<TArray extends NumberArray>(
        ArrayClass: (new (
            size: number,
            indices: Array<number>,
            values: Array<number>,
        ) => TArray),
        size: number,
        items: Iterable<[number, number]>,
    ): TArray {
        let
            indices = new Array<number>(),
            values = new Array<number>();
        for (let [index, value] of items) {
            indices.push(index);
            values.push(value);
        }
        indices.push(size); // guard
        return new ArrayClass(size, indices, values);
    }

    static _from_dense_as<TArray extends NumberArray>(
        ArrayClass: (new (
            size: number,
            indices: Array<number>,
            values: Array<number>,
        ) => TArray),
        dense: Array<number>,
    ): TArray {
        return this._from_items_as( ArrayClass,
            dense.length,
            function*(): Iterable<[number,number]> {
                for (let [index, value] of dense.entries()) {
                    if (Math.abs(value) < EPSILON)
                        continue;
                    yield [index, value];
                }
            }.call(this),
        );
    }

    _start_i(index: number): number {
        let start_index = this.indices.length - 1 - (this.size - index);
        if (start_index < 0)
            return 0;
        return start_index;
    }

    *map_items<V>(
        callbackfn: (index: number, value: number) => V,
        start?: number, end?: number,
    ): IterableIterator<V> {
        let {indices, values} = this;
        if (start === undefined)
            start = 0;
        if (end === undefined || end > this.size)
            end = this.size;
        for ( let i = this._start_i(start); i < indices.length; ++i) {
            let index = indices[i];
            if (index < start)
                continue;
            if (index >= end)
                return;
            yield callbackfn(index, values[i]);
        }
    }

    /** @yields [index, value] pairs */
    iter_items(start?: number, end?: number):
        IterableIterator<[number, number]>
    {
        return this.map_items(
            (index, value) => [index, value],
            start, end );
    }

    every_nonzero(
        callbackfn: (index: number, value: number) => boolean,
        start?: number, end?: number,
    ): boolean {
        for (let result of this.map_items(
            (index, value) => Math.abs(value) > EPSILON ?
                callbackfn(index, value) : true,
            start, end,
        ))
        {
            if (!result)
                return false;
        }
        return true;
    }

    some_nonzero(
        callbackfn: (index: number, value: number) => boolean,
        start?: number, end?: number,
    ): boolean {
        return !this.every_nonzero(
            (index, value) => !callbackfn(index, value),
            start, end,
        );
    }

    get_value(index: number, null_if_zero?: false): number
    get_value(index: number, null_if_zero: true): number | null
    get_value(index: number, null_if_zero: boolean = false): number | null {
        let [value = null_if_zero ? null : 0] =
            this.map_items((_, value) => value, index, index + 1);
        return value;
    }

    set_value(index: number, value: number | null): void {
        const {size} = this;
        if (index < 0 || index >= size)
            throw new DimensionError();
        let {indices, values} = this;
        for (let j = this._start_i(index); true; ++j) {
            let jndex = indices[j];
            if (jndex < index)
                continue;
            if (jndex > index) {
                if (value !== null) {
                    indices.splice(j, 0, index);
                    values.splice(j, 0, value);
                }
                return;
            }
            if (value !== null) {
                values[j] = value;
            } else {
                indices.splice(j, 1);
                values.splice(j, 1);
            }
            return;
        }
    }

    add_value(index: number, value: number): void {
        if (index < 0 || index >= this.size)
            throw new DimensionError();
        let {indices, values} = this;
        for (let j = this._start_i(index); true; ++j) {
            let jndex = indices[j];
            if (jndex < index)
                continue;
            if (jndex > index) {
                indices.splice(j, 0, index);
                values.splice(j, 0, value);
                return;
            }
            values[j] += value;
            return;
        }
    }

    add_items(items: Iterable<[number,number]>): void {
        const {size} = this;
        let {indices, values} = this;
        let j = 0, jndex = indices[j], prev_jndex = -1;
        let leftover_items = new Array<[number,number]>();
        for (let [index, value] of items) {
            if (index < 0 || index >= size)
                throw new DimensionError();
            while (jndex < index) {
                prev_jndex = jndex;
                jndex = indices[++j];
            }
            if (jndex > index) {
                if (prev_jndex < index) {
                    indices.splice(j, 0, index); jndex = index;
                    values.splice(j, 0, value);
                    continue;
                }
            } else {
                values[j] += value;
                continue;
            }
            leftover_items.push([index, value]);
        }
        for (let [index, value] of leftover_items) {
            this.add_value(index, value);
        }
    }

    add_from(
        array: NumberArray,
        indices: {
            start?: number,
            end?: number,
            map?: (index: number) => number | null,
        } = {},
    ): void {
        let unmapped_items = array.iter_items(indices.start, indices.end)
        if (indices.map === undefined)
            return this.add_items(unmapped_items);
        let index_map = indices.map;
        this.add_items(function*(): Iterable<[number,number]> {
            for (let [index, value] of unmapped_items) {
                let new_index = index_map(index);
                if (new_index === null)
                    continue;
                yield [new_index, value];
            }
        }.call(this));
    }

}

export class Vector extends NumberArray {
    get height() {
        return this.size;
    }

    copy(): Vector {
        return this._copy_as(Vector);
    }

    static from_items(
        height: number,
        items: Iterable<[number, number]>,
    ): Vector {
        return this._from_items_as(Vector, height, items);
    }

    static from_dense(dense: Array<number>): Vector {
        return this._from_dense_as(Vector, dense);
    }

    static zero(height: number): Vector {
        return this.from_items(height, []);
    }
}

export class CoVector extends NumberArray {

    constructor(
        width: number,
        indices: Array<number>,
        values: Array<number>,
    ) {
        super(width, indices, values);
    }

    get width() {
        return this.size;
    }

    copy(): CoVector {
        return this._copy_as(CoVector);
    }

    static from_items(
        width: number,
        items: Iterable<[number, number]>,
    ): CoVector {
        return this._from_items_as(CoVector, width, items);
    }

    static zero(width: number): CoVector {
        return this.from_items(width, []);
    }

    static from_dense(dense: Array<number>): CoVector {
        return this._from_dense_as(CoVector, dense);
    }

    apply( vector: Vector,
        {
            const_factor = 1,
            start_offset = 0,
            end_offset = 0,
        } :
        {
            const_factor?: 0 | 1,
            start_offset?: number,
            end_offset?: number,
        } = {},
    ): number {
        const n = this.width - 1, height = vector.height;
        if (n != height + start_offset + end_offset)
            throw new DimensionError();
        let indices = this.indices;
        let jndices = vector.indices;
        let result = this.get_value(n) * const_factor;
        if (indices.length <= 1 || jndices.length <= 1)
            return result;
        let start_index = start_offset > 0 ? start_offset : 0;
        for ( let
                i = this._start_i(start_index), j = 0,
                index = indices[i], jndex = jndices[j];
            true; )
        {
            if (index < jndex + start_offset) {
                ++i; index = indices[i];
                if (index >= n)
                    break;
                continue;
            }
            if (index > jndex + start_offset) {
                ++j; jndex = jndices[j];
                if (jndex >= height)
                    break;
                continue;
            }
            result += this.values[i] * vector.values[j];
            ++i; index = indices[i];
            ++j; jndex = jndices[j];
            if (index >= n || jndex >= height)
                break;
        }
        if (!isFinite(result))
            throw new Error("unreachable");
        return result;
    }

    /** @returns the change of constant element */
    shift( vector: Vector,
        apply_args?: Omit< NonNullable<Parameters<CoVector["apply"]>[1]>,
            "const_factor" >,
    ): number {
        const n = this.width - 1;
        let increase = this.apply( vector,
            Object.assign({}, apply_args, {const_factor: <0>0}) );
        this.add_value(n, increase);
        return increase;
    }

    relative_to( vector: Vector,
        apply_args?: Parameters<CoVector["shift"]>[1],
    ): CoVector {
        let relative: CoVector = this.copy();
        relative.shift(vector, apply_args);
        return relative;
    }

    scale(ratio: number | null): void {
        const {width} = this;
        if (ratio === null) {
            this.indices = [width];
            this.values = [];
            return;
        }
        let {values} = this;
        for (let i = 0; i < values.length; ++i) {
            values[i] *= ratio;
        }
    }

    /**
     * @param eliminator
     * @param symmetrical
     *   facilitates symmetrical elimination of quadratic matrix
     * @param eliminate_half
     *   facilitates symmetrical elimination of quadratic matrix
     */
    eliminate_with(
        eliminator: {
            row: CoVector,
            index: number,
            value: number,
        },
        {
            symmetrical = null,
            eliminate_half = false,
        }: {
            symmetrical?: ((index: number, increase: number) => void) | null,
            eliminate_half?: boolean,
        } = {}
    ): void {
        const {width} = this;
        if (eliminator.row.width !== width)
            throw new DimensionError();
        let {indices, values} = this;
        let {
            row: {indices: elim_indices, values: elim_values},
            index: elim_index,
        } = eliminator;
        if (elim_values.length < 1)
            throw new Error("Eliminator is a zero row");
        if (elim_index < 0 || elim_index >= width)
            throw new DimensionError();
        let ratio: number | null = null;
        for (let i = this._start_i(elim_index); i < indices.length; ++i) {
            let index = indices[i];
            if (index < elim_index)
                continue;
            if (index > elim_index)
                return;
            let value = values[i];
            if (Math.abs(value) < EPSILON) {
                indices.splice(i, 1);
                values.splice(i, 1);
                return;
            }
            ratio = -value / eliminator.value;
            break;
        }
        if (ratio === null)
            return;
        if (!isFinite(ratio))
            throw new Error("unreachable");
        if (eliminate_half)
            ratio /= 2;
        let increases = new Array<[number, number]>();
        for (let j = 0, i = 0, index = elim_indices[i]; true;) {
            let jndex = indices[j];
            if (jndex < index) {
                ++j; continue;
            }
            let increase = elim_values[i] * ratio;
            increase: {
                if (jndex > index) {
                    indices.splice(j, 0, index);
                    values.splice(j, 0, increase);
                    ++j; break increase;
                } else if (jndex === elim_index) {
                    indices.splice(j, 1);
                    values.splice(j, 1);
                    break increase;
                } else {
                    values[j] += increase;
                    ++j; break increase;
                }
            }
            if (symmetrical !== null)
                increases.push([index, increase]);
            ++i; index = elim_indices[i];
            if (index >= width)
                break;
        }
        if (symmetrical !== null) {
            for (let [index, increase] of increases)
                symmetrical(index, increase);
        }
    }

    find_extreme(
        start?: number,
        end?: number,
        {
            direction = "abs",
            value_sign = "any",
        }: {
            direction?: "min" | "max" | "abs",
            value_sign?: 1 | -1 | "any",
        } = {},
    ): {index: number, value: number} | null {
        let extreme_comparator: number = 0,
            extreme: {index: number, value: number} | null = null;
        iterate_values:
        for (let [index, value] of this.iter_items(start, end)) {
            switch (value_sign) {
                case "any":
                    break;
                case  1:
                    if (value <  EPSILON) continue iterate_values;
                    break;
                case -1:
                    if (value > -EPSILON) continue iterate_values;
                    break;
                default:
                    throw new Error("unreachable");
            }
            is_more_exreme: {
                switch (direction) {
                    case "abs":
                        let comparator = Math.abs(value);
                        if (comparator < extreme_comparator + EPSILON)
                            continue iterate_values;
                        extreme_comparator = comparator;
                        break is_more_exreme;
                    case "max":
                        if (value < extreme_comparator + EPSILON)
                            continue iterate_values;
                        extreme_comparator = value;
                        break is_more_exreme;
                    case "min":
                        if (value > extreme_comparator - EPSILON)
                            continue iterate_values;
                        extreme_comparator = value;
                        break is_more_exreme;
                }
            }
            extreme = {index, value};
        }
        return extreme;
    }

    scale_extreme(
        start?: number,
        end?: number,
        {
            direction = "abs",
            value_sign = "any",
            preserve_sign = false,
        }: {
            direction?: "min" | "max" | "abs",
            value_sign?: 1 | -1 | "any",
            preserve_sign?: boolean,
        } = {},
    ): {index: number, value: number} | null {
        let extreme_info = this.find_extreme( start, end,
            {direction: direction, value_sign: value_sign} );
        if (extreme_info === null)
            return null;
        let {index: extreme_index, value: extreme_value} = extreme_info;
        let extreme_replacement = preserve_sign ?
            (extreme_value > 0 ? +1 : -1) : 1;
        let ratio = extreme_replacement / extreme_value;
        if (!isFinite(ratio))
            throw new Error("unreachable");
        let {indices, values} = this;
        for (let i = 0; i < values.length; ++i) {
            if (indices[i] === extreme_index) {
                values[i] = extreme_replacement;
            } else {
                values[i] *= ratio;
            }
        }
        return {index: extreme_index, value: extreme_replacement};
    }

    scale_index(
        index: number,
        {preserve_sign = false}: {preserve_sign?: boolean} = {},
    ): {value: number} {
        let value = this.get_value(index);
        if (Math.abs(value) < EPSILON)
            throw new Error("unreachable");
        let replacement = preserve_sign ?
            (value > 0 ? +1 : -1) : 1;
        let ratio = replacement / value;
        if (!isFinite(ratio))
            throw new Error("unreachable");
        let {indices, values} = this;
        for (let i = 0; i < values.length; ++i) {
            if (indices[i] === index) {
                values[i] = replacement;
            } else {
                values[i] *= ratio;
            }
        }
        return {value: replacement};
    }

}

export class Matrix {
    width: number;
    rows: Array<CoVector>;

    constructor(
        width: number,
        rows: Array<CoVector>,
    ) {
        this.width = width;
        this.rows = rows;
    }

    get height(): number {
        return this.rows.length;
    }

    _copy_as<TMatrix extends Matrix>(
        MatrixClass: (new (
            width: number,
            rows: Array<CoVector>,
        ) => TMatrix),
    ): TMatrix {
        return new MatrixClass( this.width,
            this.rows.map(row => row.copy()) );
    }

    copy(): Matrix {
        return this._copy_as(Matrix);
    }

    static _from_dense_as<TMatrix extends Matrix>(
        MatrixClass: (new (
            width: number,
            rows: Array<CoVector>,
        ) => TMatrix),
        width: number,
        dense: Array<Array<number>>,
    ): TMatrix {
        return new MatrixClass( width,
            dense.map(dense_row => CoVector.from_dense(dense_row)),
        );
    }

    static from_dense(width: number, dense: Array<Array<number>>): Matrix {
        return this._from_dense_as(Matrix, width, dense);
    }

    as_dense(): Array<Array<number>> {
        let dense = this.rows.map(row => row.as_dense());
        return dense;
    }

    static _zero_as<TMatrix extends Matrix>(
        MatrixClass: (new (
            width: number,
            rows: Array<CoVector>,
        ) => TMatrix),
        width: number, height: number
    ): TMatrix {
        let rows = new Array<CoVector>();
        for (let i = 0; i < height; ++i) {
            rows.push(CoVector.zero(width));
        }
        return new MatrixClass(width, rows);
    }
    static zero(width: number, height: number): Matrix {
        return this._zero_as(Matrix, width, height);
    }

    set_value(
        row_index: number, col_index: number,
        value: number | null,
    ): void {
        if (row_index < 0 || row_index >= this.height)
            throw new DimensionError();
        this.rows[row_index].set_value(col_index, value);
    }

    add_value(
        row_index: number, col_index: number,
        value: number,
    ): void {
        if (row_index < 0 || row_index >= this.height)
            throw new DimensionError();
        this.rows[row_index].add_value(col_index, value);
    }

    add_from(
        matrix: Matrix,
        row_indices: {
            start?: number,
            end?: number,
            map?: (index: number) => number | null,
        } = {},
        col_indices: {
            start?: number,
            end?: number,
            map?: (index: number) => number | null,
        } = {},
    ): void {
        let {start: row_start, end: row_end, map: row_index_map} = row_indices;
        if (row_start === undefined || row_start < 0)
            row_start = 0;
        if (row_end === undefined || row_end > matrix.height)
            row_end = matrix.height;
        for (let row_index = row_start; row_index < row_end; ++row_index) {
            let new_row_index = row_index_map === undefined ?
                row_index : row_index_map(row_index);
            if (new_row_index === null)
                continue;
            if (new_row_index < 0 || new_row_index >= this.height)
                throw new DimensionError();
            this.rows[new_row_index].add_from(
                matrix.rows[row_index], col_indices );
        }
    }

    eliminate_with(
        eliminator: {
            row: CoVector,
            index: number,
            value: number,
            is_within?: boolean,
        },
    ): void {
        for (let row of this.rows) {
            if (eliminator.is_within && row == eliminator.row)
                continue;
            row.eliminate_with(eliminator);
        }
    }

}

export class QuadraticMatrix extends Matrix {

    copy(): QuadraticMatrix {
        return this._copy_as(QuadraticMatrix);
    }

    static from_dense(width: number | undefined, dense: Array<Array<number>>):
        QuadraticMatrix
    {
        if (width === undefined)
            width = dense.length;
        return this._from_dense_as(QuadraticMatrix, width, dense);
    }

    static zero(width: number): QuadraticMatrix {
        return this._zero_as(QuadraticMatrix, width, width);
    }

    check_symmetric(): void {
        const {width, height} = this;
        if (width !== height)
            throw new Error("Matrix shape is not symmetric");
        if (this.rows.some(row => row.width != width))
            throw new Error("Row shape is invalid");
        let dense = this.as_dense();
        for (let i = 0; i < width; ++i) {
            for (let j = 0; j < width; ++j) {
                if (Math.abs(dense[i][j] - dense[j][i]) > EPSILON)
                    throw new Error("Matrix is not symmetric");
            }
        }
    }

    shift( vector: Vector,
        apply_args?: Parameters<CoVector["shift"]>[1],
    ): void {
        const n = this.width - 1;
        let shifts = new Array<[number,number]>();
        for (let [index, row] of this.rows.entries()) {
            let shift = row.shift(vector, apply_args);
            if (Math.abs(shift) > EPSILON)
                shifts.push([index, shift]);
        }
        let last_row = this.rows[n];
        last_row.add_items(shifts);
        last_row.shift(vector, apply_args);
    }

    relative_to(vector: Vector): QuadraticMatrix {
        let relative = this.copy();
        relative.shift(vector);
        return relative;
    }

    eliminate_with(
        eliminator: {
            row: CoVector,
            index: number,
            value: number,
            is_within?: boolean,
        },
    ): void {
        const elim_index = eliminator.index;
        if (eliminator.is_within) {
            if (this.rows[eliminator.index] !== eliminator.row)
                throw new Error(
                    "Elimination from within the quadratic matrix " +
                    "can only be done by diagonal element" );
            for (let [index, row] of this.rows.entries()) {
                if (index === elim_index)
                    continue;
                row.eliminate_with(eliminator, {
                    symmetrical: (jndex, increase) => {
                        switch (jndex) {
                            case elim_index:
                                eliminator.row.set_value(index, null);
                                break;
                            case index:
                                return;
                            default:
                                this.add_value(jndex, index, increase);
                        }
                    },
                });
            }
            return;
        }
        this.rows[elim_index].eliminate_with(eliminator, {
            symmetrical: (jndex, increase) => {
                if (jndex === elim_index)
                    return;
                this.add_value(jndex, elim_index, increase);
            },
            eliminate_half: true,
        });
        this.rows[elim_index].scale(null);
        for (let [index, row] of this.rows.entries()) {
            row.eliminate_with(eliminator, {
                symmetrical: (jndex, increase) => {
                    if (jndex === elim_index)
                        return;
                    this.add_value(jndex, index, increase);
                },
            });
        }
    }

    get_definiteness(): {
        positive: number, negative: number, zero: number,
        is_positive: boolean, is_semipositive: boolean,
    } {
        return this.copy()._reduce_and_get_definiteness();
    }

    _reduce_and_get_definiteness(): {
        positive: number, negative: number, zero: number,
        is_positive: boolean, is_semipositive: boolean,
    } {
        const n = this.width - 1;
        let definiteness = {
            positive: 0,
            negative: 0,
            zero: 0,
        }
        for (let index = 0; index < n; ++index) {
            let row = this.rows[index];
            let value = row.get_value(index);
            if (Math.abs(value) < EPSILON) {
                let leader_search = row.find_extreme(0, n, {
                    direction: "abs", value_sign: "any",
                });
                if (leader_search === null) {
                    definiteness.zero += 1;
                    continue;
                }
                if (leader_search.index === index) {
                    throw new Error("unreachable");
                }
                let leader_index = leader_search.index;
                let leader_value = leader_search.value;
                let items = [...(this.rows[leader_index]
                    .map_items<[number,number]>( (i, v) =>
                        [i, i === index ? 1/2 : v / leader_value / 2]
                    ) )];
                row.add_items(items);
                for (let [i, value] of items) {
                    this.rows[i].add_value(index, value);
                }
                row.set_value(index, value = 1);
            }
            this.eliminate_with({row, index, value, is_within: true});
            if (value > 0) {
                definiteness.positive += 1;
            } else {
                definiteness.negative += 1;
            }
        }
        return Object.assign(definiteness, {
            is_semipositive:
                definiteness.negative === 0,
            is_positive:
                definiteness.negative === 0 && definiteness.zero === 0,
        });
    }

}

export class GaussianReductor {
    width: number;
    matrix: Matrix;
    consistent: boolean;
    /** @property maps matrix row indices to their leader column indices */
    row_leaders: Array<number | null>;
    column_info: Array<{eliminated_by: number}|{image_index: number}>;
    image_width: number;
    image_column_map: Array<number>;

    constructor(matrix: Matrix) {
        const width = this.width = matrix.width - 1;
        this.matrix = matrix;
        this.consistent = true;
        let row_leaders = this.row_leaders =
            new Array<number | null>(this.matrix.height);
        let column_info = this.column_info = new Array<(
            {eliminated_by: number}|{image_index: number}
        )>(this.width);
        let eliminated_count = 0;
        for (let [index, row] of matrix.rows.entries()) {
            let leading_search = row.scale_extreme(0, width,
                {direction: "abs", value_sign: "any"} );
            let leading_index: number | null = row_leaders[index] =
                (leading_search === null) ? null : leading_search.index;
            if (leading_index === null) {
                let row_constant = row.get_value(width);
                if (Math.abs(row_constant) > EPSILON) {
                    this.consistent = false;
                }
                continue;
            }
            matrix.eliminate_with({
                row, index: leading_index, value: 1, is_within: true });
            column_info[leading_index] = {eliminated_by: index};
            eliminated_count += 1;
        }
        let image_width = this.image_width = width - eliminated_count;
        let image_column_map = this.image_column_map = new Array<number>();
        let image_index = 0;
        for (let i = 0; i < width; ++i) {
            if (column_info[i] !== undefined)
                continue;
            column_info[i] = {image_index};
            image_column_map[image_index] = i;
            ++image_index;
        }
        if (image_index != image_width)
            throw new Error("unreachable");
    }

    reduce_affine_linear(
        this: GaussianReductor & {consistent: true},
        affine_form: CoVector,
    ): CoVector {
        if (!this.consistent)
            throw new Error(
                "Reduction impossible: relations were inconsistent" );
        const {width, image_width} = this;
        if (affine_form.width != width + 1)
            throw new DimensionError();
        for (let [row_index, row_leader] of this.row_leaders.entries()) {
            if (row_leader === null)
                continue;
            let row = this.matrix.rows[row_index];
            affine_form.eliminate_with({row, index: row_leader, value: 1});
        }
        return CoVector.from_items( image_width + 1,
            affine_form.map_items((index, value) => {
                if (index >= width)
                    return [image_width, value];
                let col_info = this.column_info[index];
                if ("eliminated_by" in col_info)
                    throw new Error("unreachable");
                return [col_info.image_index, value];
            }) );
    }

    reduce_affine_quadratic(
        this: GaussianReductor & {consistent: true},
        affine_form: QuadraticMatrix,
    ): QuadraticMatrix {
        if (!this.consistent)
            throw new Error(
                "Reduction impossible: relations were inconsistent" );
        const {width, image_width} = this;
        if (affine_form.width != width + 1)
            throw new DimensionError();
        for (let [row_index, row_leader] of this.row_leaders.entries()) {
            if (row_leader === null)
                continue;
            let row = this.matrix.rows[row_index];
            affine_form.eliminate_with({row, index: row_leader, value: 1});
        }
        return new QuadraticMatrix( image_width + 1,
            affine_form.rows
                .filter( (_, index) => index >= width ||
                    "image_index" in this.column_info[index] )
                .map(row => CoVector.from_items( this.image_width + 1,
                    row.map_items((index, value) => {
                        if (index >= width)
                            return [image_width, value];
                        let col_info = this.column_info[index];
                        if ("eliminated_by" in col_info)
                            throw new Error("unreachable");
                        return [col_info.image_index, value];
                    }) ))
        );
    }

    recover_vector( this: GaussianReductor & {consistent: true},
        vector: Vector ): Vector
    {
        if (!this.consistent)
            throw new Error(
                "Recover impossible: relations were inconsistent" );
        if (vector.height != this.image_width)
            throw new DimensionError();
        let new_vector = Vector.from_items( this.width,
            function*(this: GaussianReductor): Iterable<[number,number]> {
                for (let [index, value] of vector.iter_items()) {
                    yield [this.image_column_map[index], value];
                };
            }.call(this) );
        for (let [row_index, row_leader] of this.row_leaders.entries()) {
            if (row_leader === null)
                continue;
            let row = this.matrix.rows[row_index];
            new_vector.add_value(row_leader, -row.apply(new_vector));
        }
        return new_vector;
    }
}

type ConsistentReductor = GaussianReductor & {consistent: true};
type InconsistentReductor = GaussianReductor & {consistent: false};
export type MaybeConsistentReductor =
    ConsistentReductor | InconsistentReductor;

}

}