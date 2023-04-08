namespace Algebra {

class LinearExpression {
    coefficients: {[name: string]: number}
    constant: number;

    constructor(
        coefficients: {[name: string]: number},
        constant: number
    ) {
        this.coefficients = coefficients;
        this.constant = constant;
    }

    negate(): LinearExpression {
        return this.scale(-1);
    }
    plus(other: LinearExpression): LinearExpression {
        let coefficients = Object.fromEntries(
            Object.entries(this.coefficients) );
        for (let [name, value] of Object.entries(other.coefficients)) {
            coefficients[name] = (coefficients[name] || 0) + value;
        }
        return new LinearExpression( coefficients,
            this.constant + other.constant );
    }
    scale(value: number): LinearExpression {
        return new LinearExpression(
            Object.fromEntries(
                Object.entries(this.coefficients).map(
                    ([name, coefficient]) => [name, value * coefficient] )
            ),
            value * this.constant,
        );
    }

    as_constant(): number | null {
        if (Object.values(this.coefficients).length == 0) {
            return this.constant;
        }
        return null;
    }

    static from_ast(ast: ASTNode): LinearExpression {
        if (ast instanceof ASTConstant) {
            return new LinearExpression({}, ast.value);
        }
        if (ast instanceof ASTVariable) {
            return new LinearExpression({[ast.name]: 1}, 0);
        }
        if (ast instanceof ASTAdd) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right) );
        }
        if (ast instanceof ASTSub) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right).negate() );
        }
        if (ast instanceof ASTNeg) {
            return this.from_ast(ast.value).negate();
        }
        if (ast instanceof ASTMul) {
            let
                left = this.from_ast(ast.left),
                right = this.from_ast(ast.right);
            let
                left_const = left.as_constant(),
                right_const = right.as_constant();
            if (left_const !== null) {
                return right.scale(left_const);
            }
            if (right_const !== null) {
                return left.scale(right_const);
            }
            throw new Error("Cannot build any quadratic expression");
        }
        if (ast instanceof ASTDiv) {
            let
                left = this.from_ast(ast.left),
                right = this.from_ast(ast.right);
            let
                right_const = right.as_constant();
            if (right_const === null) {
                throw new Error("Cannot build a fractional expression");
            }
            if (Math.abs(right_const) < EPSILON) {
                throw new Error("Cannot divide by zero");
            }
            return left.scale(1 / right_const);
        }
        throw new Error("Unrecognized AST node ");
    }

    static from_tokens(tokens: Array<Token>): LinearExpression {
        return this.from_ast(ASTNode.parse(tokens));
    }

    *get_variables(): IterableIterator<string> {
        yield* Object.keys(this.coefficients);
    }

    as_sparse_covector(n: number, indices: {[name: string]: number}):
        Sparse.CoVector
    {
        let covector = Sparse.CoVector.zero(n + 1);
        for (let [name, value] of Object.entries(this.coefficients)) {
            covector.set_value(indices[name], value);
        }
        if (Math.abs(this.constant) > 0)
            covector.set_value(n, this.constant);
        return covector;
    }
}

class LinearRelation {
    left: LinearExpression;
    relation: "eq" | "geq";

    constructor(
        left: LinearExpression,
        relation: "eq" | "geq" | "leq",
    ) {
        if (relation == "leq") {
            left = left.negate();
            relation = "geq";
        }
        this.left = left;
        this.relation = relation;
    }

    static from_left_right(
        left: LinearExpression,
        relation: "eq" | "geq" | "leq",
        right: LinearExpression,
    ): LinearRelation {
        return new LinearRelation(left.plus(right.negate()), relation);
    }

    static from_ast(ast: ASTNode): LinearRelation {
        if (ast instanceof ASTGEq) {
            let
                left = LinearExpression.from_ast(ast.left),
                right = LinearExpression.from_ast(ast.right);
            return new LinearRelation(left.plus(right.negate()), "geq");
        }
        if (ast instanceof ASTLEq) {
            let
                left = LinearExpression.from_ast(ast.left),
                right = LinearExpression.from_ast(ast.right);
            return new LinearRelation(left.plus(right.negate()), "leq");
        }
        if (ast instanceof ASTXEq) {
            let
                left = LinearExpression.from_ast(ast.left),
                right = LinearExpression.from_ast(ast.right);
            return new LinearRelation(left.plus(right.negate()), "eq");
        }
        throw new Error("Unrecognized AST node");
    }

    static from_tokens(tokens: Array<Token>): LinearRelation {
        return this.from_ast(ASTNode.parse(tokens));
    }

    *get_variables(): IterableIterator<string> {
        yield* this.left.get_variables();
    }

    as_sparse_covector(n: number, indices: {[name: string]: number}):
        [Sparse.CoVector, "eq" | "geq"]
    {
        return [
            this.left.as_sparse_covector(n, indices),
            this.relation
        ];
    }
}

export type Token = string | number;
const enum ExprLevel {
    GROUP = 10,
    REL = 20,
    ADD = 30,
    MUL = 40,
    UNARY = 50,
};
const enum Operation {
    GEQ = 20,
    LEQ = 21,
    XEQ = 22,
    ADD = 30,
    SUB = 31,
    MUL = 40,
    DIV = 41,
};
type RelOp = Operation.GEQ | Operation.LEQ | Operation.XEQ;
type AddOp = Operation.ADD | Operation.SUB;
type MulOp = Operation.MUL | Operation.DIV;
type UnaryOp = AddOp;

export class ASTNode {
    static parse(
        tokens: Array<Token>,
    ): ASTNode {
        type Context = (
            {level: ExprLevel.GROUP, node: null, op: null} |
            {level: ExprLevel.REL, node: ASTNode, op: RelOp | null} |
            {level: ExprLevel.ADD, node: ASTNode, op: AddOp | null} |
            {level: ExprLevel.MUL, node: ASTNode, op: MulOp | null} |
            {level: ExprLevel.UNARY, node: null, op: UnaryOp}
        );
    
        let contexts: Array<Context> = new Array();
        contexts.push({level: ExprLevel.GROUP, node: null, op: null})
        let reduce_to_mul = (context: Context): Context & (
            {level: ExprLevel.GROUP} |
            {level: ExprLevel.REL} |
            {level: ExprLevel.ADD} |
            {level: ExprLevel.MUL}
        ) => {
            let unary_context = context;
            if (unary_context.level === ExprLevel.GROUP)
                return unary_context;
            if (unary_context.level === ExprLevel.REL) {
                if (unary_context.op !== null)
                    throw new Error("Relation not followed by anything");
                return unary_context;
            }
            if (unary_context.level === ExprLevel.ADD) {
                if (unary_context.op !== null)
                    throw new Error("Operator not followed by anything");
                return unary_context;
            }
            if (unary_context.level === ExprLevel.MUL) {
                if (unary_context.op !== null)
                    throw new Error("Operator not followed by anything");
                return unary_context;
            }
            throw new Error("Operator not followed by anything");
        }
        let reduce_to_add = (context: Context): Context & (
            {level: ExprLevel.GROUP} |
            {level: ExprLevel.REL} |
            {level: ExprLevel.ADD}
        ) => {
            let mul_context = reduce_to_mul(context);
            if (mul_context.level === ExprLevel.GROUP)
                return mul_context;
            if (mul_context.level === ExprLevel.REL) {
                if (mul_context.op !== null)
                    throw new Error("Relation not followed by anything");
                return mul_context;
            }
            if (mul_context.level === ExprLevel.ADD) {
                if (mul_context.op !== null)
                    throw new Error("Operator not followed by anything");
                return mul_context;
            }
            let {node, op} = mul_context;
            if (op !== null)
                throw new Error("Operator not followed by anything");
            contexts.shift();
            [context] = contexts;
            if (context.level !== ExprLevel.ADD) {
                let add_context: Context & {level: ExprLevel.ADD} =
                    {level: ExprLevel.ADD, node, op: null};
                contexts.unshift(add_context);
                return add_context;
            }
            let add_node: ASTNode;
            switch (context.op) {
                case null:
                    throw new Error("unreachable");
                case Operation.ADD:
                    add_node = new ASTAdd(context.node, node);
                    break;
                case Operation.SUB:
                    add_node = new ASTSub(context.node, node);
                    break;
            }
            let add_context: Context & {level: ExprLevel.ADD} =
                {level: ExprLevel.ADD, node: add_node, op: null};
            contexts.shift();
            contexts.unshift(add_context);
            return add_context;
        }
        let reduce_to_rel = (context: Context): Context & (
            {level: ExprLevel.GROUP} |
            {level: ExprLevel.REL}
        ) => {
            let add_context = reduce_to_add(context);
            if (add_context.level === ExprLevel.GROUP)
                return add_context;
            if (add_context.level === ExprLevel.REL) {
                if (add_context.op !== null)
                    throw new Error("Relation not followed by anything");
                return add_context;
            }
            let {node, op} = add_context;
            if (op !== null)
                throw new Error("Operator not followed by anything");
            contexts.shift();
            [context] = contexts;
            if (context.level !== ExprLevel.REL) {
                let rel_context: Context & {level: ExprLevel.REL} =
                    {level: ExprLevel.REL, node, op: null};
                contexts.unshift(rel_context);
                return rel_context;
            }
            let rel_node: ASTNode;
            switch (context.op) {
                case null:
                    throw new Error("unreachable");
                case Operation.GEQ:
                    rel_node = new ASTGEq(context.node, node);
                    break;
                case Operation.LEQ:
                    rel_node = new ASTLEq(context.node, node);
                    break;
                case Operation.XEQ:
                    rel_node = new ASTXEq(context.node, node);
                    break;
            }
            let rel_context: Context & {level: ExprLevel.REL} =
                {level: ExprLevel.REL, node: rel_node, op: null};
            contexts.shift();
            contexts.unshift(rel_context);
            return rel_context;
        };
        iterate_tokens:
        for (let token of tokens) {
            let [context] = contexts;
            let value: ASTNode;
            switch (token) {
                case ">=": {
                    let rel_context = reduce_to_rel(context);
                    if (rel_context.level === ExprLevel.GROUP)
                        throw new Error("Relation not preceded by anything")
                    rel_context.op = Operation.GEQ;
                    continue iterate_tokens;
                }
                case "<=": {
                    let rel_context = reduce_to_rel(context);
                    if (rel_context.level === ExprLevel.GROUP)
                        throw new Error("Relation not preceded by anything")
                    rel_context.op = Operation.LEQ;
                    continue iterate_tokens;
                }
                case "==": {
                    let rel_context = reduce_to_rel(context);
                    if (rel_context.level === ExprLevel.GROUP)
                        throw new Error("Relation not preceded by anything")
                    rel_context.op = Operation.XEQ;
                    continue iterate_tokens;
                }
                case "+" : {
                    if (
                        context.level === ExprLevel.GROUP ||
                        context.level === ExprLevel.REL ||
                        context.op !== null
                    ) {
                        contexts.unshift({
                            level: ExprLevel.UNARY,
                            node: null,
                            op: Operation.ADD,
                        });
                        continue iterate_tokens;
                    }
                    let add_context = reduce_to_add(context);
                    if (
                        add_context.level === ExprLevel.GROUP ||
                        add_context.level === ExprLevel.REL )
                        throw new Error("Operation not preceded by anything")
                    add_context.op = Operation.ADD;
                    continue iterate_tokens;
                }
                case "-" : {
                    if (
                        context.level === ExprLevel.GROUP ||
                        context.level === ExprLevel.REL ||
                        context.op !== null
                    ) {
                        contexts.unshift({
                            level: ExprLevel.UNARY,
                            node: null,
                            op: Operation.SUB,
                        });
                        continue iterate_tokens;
                    }
                    let add_context = reduce_to_add(context);
                    if (
                        add_context.level === ExprLevel.GROUP ||
                        add_context.level === ExprLevel.REL )
                        throw new Error("Operation not preceded by anything")
                    add_context.op = Operation.SUB;
                    continue iterate_tokens;
                }
                case "*" : {
                    let mul_context = reduce_to_mul(context);
                    if (
                        mul_context.level === ExprLevel.GROUP ||
                        mul_context.level === ExprLevel.REL ||
                        mul_context.level === ExprLevel.ADD )
                        throw new Error("Operation not preceded by anything")
                    mul_context.op = Operation.MUL;
                    continue iterate_tokens;
                }
                case "/" : {
                    let mul_context = reduce_to_mul(context);
                    if (
                        mul_context.level === ExprLevel.GROUP ||
                        mul_context.level === ExprLevel.REL ||
                        mul_context.level === ExprLevel.ADD )
                        throw new Error("Operation not preceded by anything")
                    mul_context.op = Operation.DIV;
                    continue iterate_tokens;
                }
                case "(" : switch (context.level) {
                    case ExprLevel.GROUP:
                        contexts.unshift({ level: ExprLevel.GROUP,
                            node: null, op: null });
                        continue iterate_tokens;
                    case ExprLevel.REL:
                    case ExprLevel.ADD:
                    case ExprLevel.MUL:
                    case ExprLevel.UNARY:
                        if (context.op === null)
                            throw new Error(
                                "Value is not preceded by operation" )
                        contexts.unshift({ level: ExprLevel.GROUP,
                            node: null, op: null });
                        continue iterate_tokens;
                    default:
                        throw new Error("unreachable");
                }
                case ")" : {
                    let rel_context = reduce_to_rel(context);
                    if (rel_context.level === ExprLevel.GROUP)
                        throw new Error("Empty parentheses pair");
                    contexts.shift();
                    let opening_context = contexts.shift();
                    if (opening_context === undefined || contexts.length === 0)
                        throw new Error("Unmatched closing parenthesis");
                    if (opening_context.level !== ExprLevel.GROUP)
                        throw new Error("unreachable");
                    value = rel_context.node;
                    [context] = contexts;
                    break;
                }
                default: {
                    if (typeof token === "number") {
                        value = new ASTConstant(token);
                    } else {
                        value = new ASTVariable(token);
                    }
                }
            }
            while (value !== null) {
                switch (context.level) {
                    case ExprLevel.GROUP:
                        contexts.unshift({ level: ExprLevel.MUL,
                            node: value, op: null });
                        continue iterate_tokens;
                    case ExprLevel.REL:
                    case ExprLevel.ADD:
                        if (context.op === null)
                            throw new Error(
                                "Value is not preceded by operation" )
                        contexts.unshift({ level: ExprLevel.MUL,
                            node: value, op: null });
                        continue iterate_tokens;
                    case ExprLevel.MUL: switch(context.op) {
                        case null:
                            throw new Error(
                                "Value is not preceded by operation" )
                        case Operation.MUL:
                            contexts.shift();
                            contexts.unshift({ level: ExprLevel.MUL, op: null,
                                node: new ASTMul(context.node, value) });
                            continue iterate_tokens;
                        case Operation.DIV:
                            contexts.shift();
                            contexts.unshift({ level: ExprLevel.MUL, op: null,
                                node: new ASTDiv(context.node, value) });
                            continue iterate_tokens;
                        default:
                            throw new Error("unreachable");
                    }
                    default: switch (context.op) {
                        case Operation.ADD:
                            contexts.shift();
                            [context] = contexts;
                            continue;
                        case Operation.SUB:
                            value = new ASTNeg(value);
                            contexts.shift();
                            [context] = contexts;
                            continue;
                        default:
                            throw new Error("unreachable");
                    }
                }
            }
        }
        let rel_context = reduce_to_rel(contexts[0]);
        contexts.shift();
        if (contexts.length > 1) {
            throw new Error("Unmatched opening parenthesis");
        }
        if (contexts.length < 1) {
            throw new Error("unreachable");
        }
        if (rel_context.level === ExprLevel.GROUP)
            throw new Error("Empty expression");
        return rel_context.node;
    }
}

class ASTConstant extends ASTNode {
    value: number;
    constructor(value: number) {
        super();
        this.value = value;
    }
}

class ASTVariable extends ASTNode {
    name: string;
    constructor(name: string) {
        super();
        this.name = name;
    }
}

class ASTUnary extends ASTNode {
    value: ASTNode;
    constructor(value: ASTNode) {
        super();
        this.value = value;
    }
}

class ASTNeg extends ASTUnary {}

class ASTBinary extends ASTNode {
    left: ASTNode;
    right: ASTNode;
    constructor(left: ASTNode, right: ASTNode) {
        super();
        this.left = left;
        this.right = right;
    }
}

class ASTGEq extends ASTBinary {}
class ASTLEq extends ASTBinary {}
class ASTXEq extends ASTBinary {}

class ASTAdd extends ASTBinary {}
class ASTSub extends ASTBinary {}

class ASTMul extends ASTBinary {}
class ASTDiv extends ASTBinary {}

export namespace Sparse {

class DimensionError extends Error {};

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

    _copy_as<TSparse extends NumberArray>(SparseClass: (new (
        size: number,
        indices: Array<number>,
        values: Array<number>,
    ) => TSparse)): TSparse {
        return new SparseClass( this.size,
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

    get_value(index: number, null_if_zero?: false): number
    get_value(index: number, null_if_zero: true): number | null
    get_value(index: number, null_if_zero: boolean = false): number | null {
        let [value = null_if_zero ? null : 0] =
            this.map_items((_, value) => value, index, index + 1);
        return value;
    }

    set_value(index: number, value: number | null): void {
        if (index < 0 || index >= this.size)
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
                values[j] += value;
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
        const width = this.width - 1, height = vector.height;
        if (width != height + start_offset + end_offset)
            throw new DimensionError();
        let indices = this.indices;
        let jndices = vector.indices;
        let result = this.get_value(width) * const_factor;
        if (indices.length == 0 || jndices.length == 0)
            return result;
        let start_index = start_offset > 0 ? start_offset : 0;
        for ( let
                i = this._start_i(start_index), j = 0,
                index = indices[i], jndex = jndices[j];
            true; )
        {
            if (index < jndex + start_offset) {
                ++i; index = indices[i];
                if (index >= width)
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
            if (index >= width || jndex >= height)
                break;
            // XXX debug
            if (!isFinite(result))
                throw new Error("unreachable");
        }
        return result;
    }

    relative_to(vector: Vector): CoVector {
        const {width} = this;
        let relative: CoVector = this.copy();
        relative.add_value(width-1, this.apply(vector, {const_factor: 0}));
        return relative;
    }

    scale(ratio: number | null): void {
        const {width} = this;
        if (ratio === null) {
            this.indices = [width];
            this.values = [];
            return;
        }
        let {indices, values} = this;
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
            objective: direction = "abs",
            objective_sign: value_sign = "any",
            preserve_sign = false,
        }: {
            objective?: "min" | "max" | "abs",
            objective_sign?: 1 | -1 | "any",
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

    copy_matrix(): Matrix {
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

}

export class QuadraticMatrix extends Matrix {

    copy_matrix(): QuadraticMatrix {
        return this._copy_as(QuadraticMatrix);
    }

    static from_dense(width: number, dense: Array<Array<number>>):
        QuadraticMatrix
    {
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

    relative_to(vector: Vector): QuadraticMatrix {
        const {width, height} = this;
        let last_column_items = new Array<[number,number]>();
        let relative = new QuadraticMatrix( this.width,
            this.rows.map((row, index) => {
                let relative_row = row.relative_to(vector);
                let last_value = relative_row.get_value(width-1, true);
                if (last_value !== null)
                    last_column_items.push([index, last_value]);
                return relative_row;
            }) );
        relative.rows[height-1] =
            CoVector.from_items(width-1, last_column_items)
                .relative_to(vector);
        return relative;
    }

    eliminate_with(
        eliminator: {
            row: CoVector,
            index: number,
            value: number,
        },
    ): void {
        let {
            row: {indices: elim_indices, values: elim_values},
            index: elim_index,
        } = eliminator;
        this.rows[elim_index].eliminate_with(eliminator, {
            symmetrical: (jndex, increase) => {
                if (jndex === elim_index)
                    return;
                this.rows[jndex].add_value(elim_index, increase);
            },
            eliminate_half: true,
        });
        this.rows[elim_index].scale(null);
        for (let [index, row] of this.rows.entries()) {
            row.eliminate_with(eliminator, {
                symmetrical: (jndex, increase) => {
                    if (jndex === elim_index)
                        return;
                    this.rows[jndex].add_value(index, increase);
                }
            });
        }
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
                {objective: "abs", objective_sign: "any"} );
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
type MaybeConsistentReductor = ConsistentReductor | InconsistentReductor;

export function test_gauss() {
    const size = 4;
    let reductor: MaybeConsistentReductor =
        new GaussianReductor(Matrix.from_dense(size+1, [
        [ 1,  1,  1,  1,  0],
        [ 1,  1,  1,  1,  0],
        [ 0,  0, 42,  0, 42],
        [ 0,  0,  0,  1,  0],
    ]));
    if (!reductor.consistent)
        return {reductor};
    let qmatrix = QuadraticMatrix.from_dense( size+1, [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]);
    qmatrix.check_symmetric();
    let qmatrix_reduced = reductor.reduce_affine_quadratic(
        qmatrix.copy_matrix() );
    qmatrix_reduced.check_symmetric();
    return {
        reductor,
        qmatrix, qmatrix_reduced,
        qmatrix_reduces_expected: [[-1, 1], [1, 0]],
    };
}

type Solution = Vector;
type ErrorReport = {error: true, description: string};


export class LPProblemSolver {
    n: number;
    k: number;
    k0: number;
    k1: number;
    point: Vector;
    row_indices: Array<(
        {tableau_col: number, variable_col?: undefined} |
        {variable_col: number, tableau_col?: undefined}
    )>;
    col_indices: {
        constraint: Array<number|null>,
        variable: Array<number|null>,
    };
    tableau: {
        constraint: Matrix,
        objective: CoVector,
    };
    result: null | Solution | ErrorReport;

    static solve(
        {
            objective,
            constraints_eq,
            constraints_geq,
        }: {
            objective: CoVector,
            constraints_eq: Matrix | null,
            constraints_geq: Matrix,
        }
    ): Solution & {error?: false} | ErrorReport {
        let reconstruct_solution: (solution: Solution) => Solution =
            (solution) => solution;
        if (constraints_eq !== null && constraints_eq.height > 0) {
            let maybe_reductor: MaybeConsistentReductor =
                new GaussianReductor(constraints_eq);
            if (!maybe_reductor.consistent)
                return { error: true,
                    description: "Equations are inconsistent" };
            let reductor = maybe_reductor;
            reconstruct_solution =
                (solution) => reductor.recover_vector(solution);
            constraints_geq = new Matrix( reductor.image_width + 1,
                constraints_geq.rows.map( row =>
                    reductor.reduce_affine_linear(row.copy()) )
            );
            objective = reductor.reduce_affine_linear(objective.copy());
        }
        let feasible_point =
            LPProblemSolver._find_feasible_point({ constraints:
                constraints_geq });
        if ("error" in feasible_point) {
            return feasible_point;
        }
        let solution = new LPProblemSolver({
            objective,
            constraints: constraints_geq,
            feasible_point,
        }).get_solution();
        if ("error" in solution)
            return solution;
        return reconstruct_solution(solution);
    }

    static solve_from_tokens(
        objective: Token[],
        target: "min",
        constraints: Array<Token[]>,
    ): {error?: false, solution: {[name: string]: number}} | ErrorReport {
        let objective_expr = LinearExpression.from_tokens(objective);
        let constraints_rels = constraints.map( tokens =>
            LinearRelation.from_tokens(tokens) );
        let
            variables = new Array<string>(),
            variable_set = new Set<string>();
        let add_variable = (variable: string) => {
            if (!variable_set.has(variable)) {
                variables.push(variable);
                variable_set.add(variable);
            }
        }
        for (let variable of objective_expr.get_variables()) {
            add_variable(variable);
        }
        for (let constraint of constraints_rels) {
            for (let variable of constraint.get_variables()) {
                add_variable(variable);
            }
        }
        let indices: {[name: string]: number} = {};
        for (let [index, name] of variables.entries()) {
            indices[name] = index;
        }
        let n = variables.length;
        let sparse_constraints = constraints_rels
            .map(constraint => constraint.as_sparse_covector(n, indices));

        let solution = LPProblemSolver.solve({
            objective: objective_expr.as_sparse_covector(n, indices),
            constraints_eq: new Matrix( n + 1,
                sparse_constraints
                    .filter(([,relation]) => relation === "eq")
                    .map(([covector]) => covector),
            ),
            constraints_geq: new Matrix( n + 1,
                sparse_constraints
                    .filter(([,relation]) => relation === "geq")
                    .map(([covector]) => covector),
            ),
        })
        if (solution?.error) {
            return solution;
        }
        if (solution.height != n) {
            throw new Error("unreachable");
        }
        let successful = solution;
        return {solution: Object.fromEntries(variables.map(
            (name, index) => [name, successful.get_value(index)] ))};
    }

    constructor(
        {
            objective,
            constraints,
            feasible_point,
        }: {
            objective: CoVector,
            constraints: Matrix,
            feasible_point: Vector,
        }
    ) {
        const n = this.n = objective.width - 1;
        const k = this.k = constraints.height;
        if (constraints.rows.some(row => row.width !== n + 1))
            throw new DimensionError();
        if (feasible_point.height !== n)
            throw new DimensionError();

        this.k0 = 0;
        this.k1 = k;
        this.point = feasible_point;
        this.row_indices = new Array<{tableau_col: number}>();
        this.col_indices = {
            constraint: new Array<number>(k),
            variable: new Array<null>(n),
        }
        this.tableau = {
            constraint: ((): Matrix => {
                let cstr_residual = new Matrix( n + 1,
                    constraints.rows.map(row => {
                        let r = row.relative_to(this.point);
                        r.scale(-1);
                        return r;
                    }));
                let tableau = Matrix.zero(k + n + 1, k);
                tableau.add_from(cstr_residual, {}, { /* col indices */
                    map: (index) => k + index,
                });
                for (let i = 0; i < k; ++i) {
                    tableau.rows[i].add_value(i, 1);
                }
                return tableau;
            })(),
            objective: ((): CoVector => {
                let obj_residual = objective.relative_to(this.point);
                let objective_tableau = CoVector.from_items(k + n + 1, []);
                objective_tableau.add_from(obj_residual, {
                    map: (index) => k + index,
                });
                return objective_tableau;
            })(),
        };
        this.col_indices.variable.fill(null);
        for (let [index, ] of constraints.rows.entries()) {
            this.row_indices.push({tableau_col: index});
            this.col_indices.constraint.push(index);
        }
        this.result = null;
    }

    get_solution(): Solution & {error?: false} | ErrorReport {
        while (this.result === null)
            this._step();
        return this.result;
    }

    static _find_feasible_point(
        {constraints}: {constraints: Matrix}
    ): Solution | ErrorReport {
        const n = constraints.width - 1;
        let point = Vector.zero(n);
        let differences = constraints.rows.map(row => row.apply(point));
        if (differences.every(d => d > -EPSILON))
            return point;
        let feasibility_initial = Vector.zero(n + 1);
        feasibility_initial.add_value(0, -Math.min(...differences));
        let feasibility_constraints = Matrix.zero(n + 2, constraints.height);
        feasibility_constraints.add_from( constraints,
            {}, {map: (index) => index + 1} );
        for (let constraint of feasibility_constraints.rows) {
            constraint.set_value(0, 1);
        }
        let feasibility_objective = CoVector.zero(n + 2);
        feasibility_objective.set_value(0, 1);
        feasibility_constraints.rows.push(feasibility_objective);
        let feasibility_solution = new LPProblemSolver({
            objective: feasibility_objective,
            constraints: feasibility_constraints,
            feasible_point: feasibility_initial,
        }).get_solution();
        if (feasibility_solution.error) {
            return feasibility_solution;
        }
        if (feasibility_solution.get_value(0) > EPSILON) {
            return {error: true, description: "Infeasible"};            
        }
        let feasible_point = Vector.zero(n);
        feasible_point.add_from( feasibility_solution,
            {start: 1, map: (index) => index - 1} );
        // XXX debug
        for (let constraint of constraints.rows) {
            if (constraint.apply(feasible_point) < -EPSILON) {
                throw new Error("Feasible point is not actually feasible");
            }
        }
        return feasible_point;
    }

    _step(): void {
        const {n, k} = this;
        let obj_tableau = this.tableau.objective;
        for (let [index, value] of obj_tableau.iter_items(k, k + n)) {
            if (Math.abs(value) < EPSILON)
                continue;
            return this._step_enter(index - k, value > 0 ? -1 : +1);
        }
        for (let [index, value] of obj_tableau.iter_items(0, k)) {
            if (value > -EPSILON)
                continue;
            return this._step_exit(index);
        }
        this.result = this.point;
    }

    _step_enter(
        entering_var_index: number,
        entering_var_sign: 1 | -1,
    ): void {
        const {n, k} = this;
        let
            entering_vector: Vector,
            constraint_vector = Vector.zero(k),
            objective_vector: number,
            entering: {index: number, cstr_index: number} | null = null,
            vector_scale: number = Infinity;
        { // find entering index
            let entering_vector_base = Vector.zero(n);
            entering_vector_base.set_value(
                entering_var_index, entering_var_sign );
            objective_vector =
                this.tableau.objective.apply( entering_vector_base,
                    {start_offset: k, const_factor: 0} );
            entering_vector = entering_vector_base.copy();
            for (let [index, index_info] of this.row_indices.entries()) {
                let row = this.tableau.constraint.rows[index];
                if (index_info.variable_col !== undefined) {
                    entering_vector.add_value(index_info.variable_col, 
                        -row.apply( entering_vector_base,
                            {start_offset: k, const_factor: 0} )
                    );
                    continue;
                }
                let entering_speed = -row.apply( entering_vector_base,
                    {start_offset: k, const_factor: 0} );
                constraint_vector.add_value(index, entering_speed);
                if (entering_speed > -EPSILON)
                    continue;
                let margin = -row.get_value(k + n);
                if (margin < -EPSILON)
                    throw new Error("unreachable");
                if (margin < EPSILON) {
                    // Bland's rule, kinda
                    if (
                        entering !== null &&
                        vector_scale === 0 &&
                        index_info.tableau_col > entering.cstr_index
                    ) {
                        continue;
                    }
                    vector_scale = 0;
                    entering = { index,
                        cstr_index: index_info.tableau_col };
                    continue;
                }
                let scale = margin / -entering_speed;
                if (scale < -EPSILON) {
                    throw new Error("unreachable");
                }
                if (scale < vector_scale - EPSILON) {
                    vector_scale = scale;
                    entering = { index,
                        cstr_index: index_info.tableau_col };
                }
            }
        }
        if (entering === null) {
            this.result = {error: true, description: "Unbounded"};
            return;
        }
        let entering_index = entering.index;
        if (!isFinite(vector_scale))
            throw new Error("unreachable");

        // shift the point
        for (let [index, value] of constraint_vector.iter_items()) {
            this.tableau.constraint.rows[index]
                .add_value(k + n, vector_scale * -value);
        }
        this.tableau.objective
            .add_value(k + n, vector_scale * objective_vector);
        this.point.add_items( entering_vector
            .map_items((index, value) => [index, vector_scale * value]) )

        let entering_row = this.tableau.constraint.rows[entering_index];
        // XXX debug
        if (Math.abs(entering_row.get_value(k + n)) > EPSILON) {
            throw new Error("unreachable")
        }
        entering_row.set_value(k + n, null);

        entering_row.scale_index(k + entering_var_index);
        this.tableau.constraint.eliminate_with({
            row: entering_row, index: k + entering_var_index, value: 1,
            is_within: true });
        this.tableau.objective.eliminate_with({
            row: entering_row, index: k + entering_var_index, value: 1 });

        this.col_indices.constraint[entering.cstr_index] = null;
        this.col_indices.variable[entering_var_index] = entering_index;
        this.row_indices[entering_index] = {variable_col: entering_var_index};

        this.k0 += 1;
        this.k1 -= 1;
    }

    _step_exit(
        leaving_cstr_index: number,
    ): void {
        const {n, k} = this;
        let leaving: {index: number, var_index: number} | null = null;
        { // find leaving row
            for (let [index, index_info] of this.row_indices.entries()) {
                if (index_info.tableau_col !== undefined) {
                    continue;
                }
                let row = this.tableau.constraint.rows[index];
                let value = row.get_value(leaving_cstr_index, true);
                if (value === null || Math.abs(value) < EPSILON)
                    continue;
                // Bland's rule, kinda
                if ( leaving !== null &&
                    index_info.variable_col > leaving.var_index
                ) {
                    continue;
                }
                leaving = { index,
                    var_index: index_info.variable_col };
                continue;
            }
        }
        if (leaving === null)
            throw new Error("unreachable");

        let leaving_index = leaving.index;
        let leaving_row = this.tableau.constraint.rows[leaving_index];
        leaving_row.scale_index(leaving_cstr_index);
        this.tableau.constraint.eliminate_with({
            row: leaving_row, index: leaving_cstr_index, value: 1,
            is_within: true });
        this.tableau.objective.eliminate_with({
            row: leaving_row, index: leaving_cstr_index, value: 1 });

        this.col_indices.constraint[leaving_cstr_index] = leaving_index;
        this.col_indices.variable[leaving.var_index] = null;
        this.row_indices[leaving_index] = {tableau_col: leaving_cstr_index};

        this.k0 -= 1;
        this.k1 += 1;
    }
}

/* QPProblemSolver draft
class QPProblemSolver {
    n: number;
    k: number;
    k0: number;
    k1: number;
    point: Vector;
    row_indices: Array<{tableau_col: number}|{residual_col: number}>;
    col_indices: {
        constraint: Array<number|null>,
        variable: Array<number|null>,
    };
    tableau: {
        constraint: Matrix,
        objective: QuadraticMatrix,
    };
    result: null | Solution | ErrorReport;

    static solve(
        {
            objective,
            constraints_eq,
            constraints_geq,
        }: {
            objective: QuadraticMatrix,
            constraints_eq: Matrix,
            constraints_geq: Matrix,
        }
    ): Solution | ErrorReport {
        if (constraints_eq.height > 0) {
            return …;
        }
        let feasible_point =
            QPProblemSolver._find_feasible_point({constraints_geq});
        if ("error" in feasible_point) {
            return feasible_point;
        }
        return new QPProblemSolver({
            objective,
            constraints: constraints_geq,
            feasible_point,
        }).get_solution();
    }

    constructor(
        {
            objective,
            constraints,
            feasible_point,
        }: {
            objective: QuadraticMatrix,
            constraints: Matrix,
            feasible_point: Vector,
        }
    ) {
        const n = this.n = objective.width - 1;
        const k = this.k = constraints.height;

        this.k0 = 0;
        this.k1 = k;
        this.point = feasible_point;
        this.row_indices = new Array<{tableau_col: number}>();
        this.col_indices = {
            constraint: new Array<number>(k),
            variable: new Array<null>(n),
        }
        this.tableau = {
            constraint: new Matrix(k + n + 1, constraints.rows.map(
                (constraint, jndex) => CoVector.from_items(k + n + 1, [
                    [jndex, 1],
                    ...constraint.map_items<[number,number]>(
                        (index, value) => [ index + k, index < n
                            ? -value
                            : -constraint.apply(this.point) ]
                    ),
                ])
            )),
            objective: new QuadraticMatrix(k + n + 1, [
                ...constraints.rows
                    .map(() => CoVector.from_items(k + n + 1, [])),
                ...(objective.relative_to(this.point).rows
                .map<CoVector>((row) => CoVector.from_items(
                    k + n + 1,
                    row.map_items<[number,number]>(
                        (index, value) => [index + k, value]
                    )
                ))),
            ]),
        };
        this.col_indices.variable.fill(null);
        for (let [index, ] of constraints.rows.entries()) {
            this.row_indices.push({tableau_col: index});
            this.col_indices.constraint.push(index);
        }
        this.result = null;
    }

    get_solution(): Solution | ErrorReport {
        while (this.result === null)
            this._step();
        return this.result;
    }

    static _find_feasible_point(
        {
            constraints_geq,
        }: {
            constraints_geq: Matrix,
        }
    ): Solution | ErrorReport {
        return …;
    }

    _step(): void {
        …;
    }
}

export function solve_qproblem() {
    return …;
}
*/

} // end namespace

} // end namespace


// replacement for js-lp-solver
var solver = {
Solve: function Solve(model: LPModel): { [s: string]: any; } {
    type Token = Algebra.Token;
    let objective: Token[] = [model.optimize];
    if (model.opType === "max") {
        objective = [-1, "*", "(", ...objective, ")"];
    } else if (model.opType !== "min") {
        throw new Error("unknown opType " + model.opType);
    }
    let constraints: Array<Token[]> = [];
    let constraint_map: {[name: string]: {[name: string]: number}} = {};
    for (let [var_name, coefficients] of Object.entries(model.variables)) {
        for (let [constr_name, coefficient] of Object.entries(coefficients)) {
            let constraint_coeffs = constraint_map[constr_name];
            if (constraint_coeffs === undefined)
                constraint_coeffs = constraint_map[constr_name] = {};
            constraint_coeffs[var_name] = coefficient;
        }
        constraints.push([var_name, ">=", 0]);
    }
    for (let [constr_name, constr_map] of Object.entries(constraint_map)) {
        let expression: Token[] = [];
        for (let [var_name, coefficient] of Object.entries(constr_map))
        {
            expression.push("+", coefficient, "*", var_name);
        }
        constraints.push([constr_name, "==", ...expression]);
        let constr_limits = model.constraints[constr_name];
        if (constr_limits === undefined)
            continue;
        if (constr_limits.equal !== undefined) {
            constraints.push([constr_name, "==", constr_limits.equal]);
            continue;
        }
        if (constr_limits.max !== undefined) {
            constraints.push([constr_name, "<=", constr_limits.max]);
        }
        if (constr_limits.min !== undefined) {
            constraints.push([constr_name, ">=", constr_limits.min]);
        }
    }
    let solution = Algebra.Sparse.LPProblemSolver.solve_from_tokens(
        objective,
        "min",
        constraints,
    );
    if (solution.error) {
        return Object.assign({feasible: false}, solution);
    }
    return Object.assign({feasible: true}, solution.solution);
}
}
