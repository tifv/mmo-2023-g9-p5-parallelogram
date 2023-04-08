namespace Algebra {

export namespace Expression {


export type Token = string | number;
export type Tokens = Array<Token>;


enum NodeLevel {
    GROUP = 10,
    REL = 20,
    ADD = 30,
    MUL = 40,
    UNARY = 50,
};
enum Operation {
    GEQ = 20,
    LEQ = 21,
    EQ = 22,
    ADD = 30,
    SUB = 31,
    MUL = 40,
    DIV = 41,
};
type RelOp = Operation.GEQ | Operation.LEQ | Operation.EQ;
type AddOp = Operation.ADD | Operation.SUB;
type MulOp = Operation.MUL | Operation.DIV;
type UnaryOp = AddOp;


const NODE_TYPE = Symbol();
type ASTNode = (
    ASTExpression | ASTRelation
);

class ASTExpressionClass {
    get [NODE_TYPE](): "expression" { return "expression"; }
};

class ASTRelationClass {
    get [NODE_TYPE](): "relation" { return "relation"; }
}

const EXPR_TYPE = Symbol();
type ASTExpression = (
    ASTConstant | ASTVariable | ASTUnary | ASTBinary
);

class ASTConstant extends ASTExpressionClass {
    get [EXPR_TYPE](): "constant" { return "constant"; }
    value: number;
    constructor(value: number) {
        super();
        this.value = value;
    }
}

class ASTVariable extends ASTExpressionClass {
    get [EXPR_TYPE](): "variable" { return "variable"; }
    name: string;
    constructor(name: string) {
        super();
        this.name = name;
    }
}

class ASTUnaryBase extends ASTExpressionClass {
    get [EXPR_TYPE](): "unary" { return "unary"; }
    value: ASTExpression;
    constructor(value: ASTNode) {
        super();
        if (value[NODE_TYPE] !== "expression")
            throw new Error("Unary operator applied to a non-expression.");
        this.value = value;
    }
}

class ASTBinaryClass extends ASTExpressionClass {
    get [EXPR_TYPE](): "binary" { return "binary"; }
    left: ASTExpression;
    right: ASTExpression;
    constructor(left: ASTNode, right: ASTNode) {
        super();
        if (
            left [NODE_TYPE] !== "expression" ||
            right[NODE_TYPE] !== "expression"
        )
            throw new Error("Binary operator applied to a non-expression.");
        this.left = left;
        this.right = right;
    }
}

const OPERATION = Symbol();

type ASTUnary = ASTNeg;

class ASTNeg extends ASTUnaryBase {
    get [OPERATION](): Operation.SUB { return Operation.SUB; }
}

type ASTBinary = ASTAdd | ASTSub | ASTMul | ASTDiv;

class ASTAdd extends ASTBinaryClass {
    get [OPERATION](): Operation.ADD { return Operation.ADD; }
};
class ASTSub extends ASTBinaryClass {
    get [OPERATION](): Operation.SUB { return Operation.SUB; }
};

class ASTMul extends ASTBinaryClass {
    get [OPERATION](): Operation.MUL { return Operation.MUL; }
};
class ASTDiv extends ASTBinaryClass {
    get [OPERATION](): Operation.DIV { return Operation.DIV; }
};


const REL_TYPE = Symbol();
type ASTRelation = ASTBinaryRelation;

class ASTBinaryRelationClass extends ASTRelationClass {
    get [REL_TYPE](): "binary" { return "binary"; }
    left: ASTExpression;
    right: ASTExpression;
    constructor(left: ASTNode, right: ASTNode) {
        super();
        if (
            left[NODE_TYPE] !== "expression" ||
            right[NODE_TYPE] !== "expression"
        )
            throw new Error("Relation applied to a non-expression.");
        this.left = left;
        this.right = right;
    }
}


type ASTBinaryRelation = ASTGEq | ASTLEq | ASTEq;

class ASTGEq extends ASTBinaryRelationClass {
    get [OPERATION](): Operation.GEQ { return Operation.GEQ; }
};
class ASTLEq extends ASTBinaryRelationClass {
    get [OPERATION](): Operation.LEQ { return Operation.LEQ; }
};
class ASTEq extends ASTBinaryRelationClass {
    get [OPERATION](): Operation.EQ  { return Operation.EQ; }
};


function parse_tokens(tokens: Tokens): ASTNode {
    type Context = (
        {level: NodeLevel.GROUP, node: null, op: null} |
        {level: NodeLevel.REL, node: ASTNode, op: RelOp | null} |
        {level: NodeLevel.ADD, node: ASTNode, op: AddOp | null} |
        {level: NodeLevel.MUL, node: ASTNode, op: MulOp | null} |
        {level: NodeLevel.UNARY, node: null, op: UnaryOp}
    );

    let contexts: Array<Context> = new Array();
    contexts.push({level: NodeLevel.GROUP, node: null, op: null})
    let reduce_to_mul = (context: Context): Context & (
        {level: NodeLevel.GROUP} |
        {level: NodeLevel.REL} |
        {level: NodeLevel.ADD} |
        {level: NodeLevel.MUL}
    ) => {
        let unary_context = context;
        if (unary_context.level === NodeLevel.GROUP)
            return unary_context;
        if (unary_context.level === NodeLevel.REL) {
            if (unary_context.op !== null)
                throw new Error("Relation not followed by anything");
            return unary_context;
        }
        if (unary_context.level === NodeLevel.ADD) {
            if (unary_context.op !== null)
                throw new Error("Operator not followed by anything");
            return unary_context;
        }
        if (unary_context.level === NodeLevel.MUL) {
            if (unary_context.op !== null)
                throw new Error("Operator not followed by anything");
            return unary_context;
        }
        throw new Error("Operator not followed by anything");
    }
    let reduce_to_add = (context: Context): Context & (
        {level: NodeLevel.GROUP} |
        {level: NodeLevel.REL} |
        {level: NodeLevel.ADD}
    ) => {
        let mul_context = reduce_to_mul(context);
        if (mul_context.level === NodeLevel.GROUP)
            return mul_context;
        if (mul_context.level === NodeLevel.REL) {
            if (mul_context.op !== null)
                throw new Error("Relation not followed by anything");
            return mul_context;
        }
        if (mul_context.level === NodeLevel.ADD) {
            if (mul_context.op !== null)
                throw new Error("Operator not followed by anything");
            return mul_context;
        }
        let {node, op} = mul_context;
        if (op !== null)
            throw new Error("Operator not followed by anything");
        contexts.shift();
        [context] = contexts;
        if (context.level !== NodeLevel.ADD) {
            let add_context: Context & {level: NodeLevel.ADD} =
                {level: NodeLevel.ADD, node, op: null};
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
        let add_context: Context & {level: NodeLevel.ADD} =
            {level: NodeLevel.ADD, node: add_node, op: null};
        contexts.shift();
        contexts.unshift(add_context);
        return add_context;
    }
    let reduce_to_rel = (context: Context): Context & (
        {level: NodeLevel.GROUP} |
        {level: NodeLevel.REL}
    ) => {
        let add_context = reduce_to_add(context);
        if (add_context.level === NodeLevel.GROUP)
            return add_context;
        if (add_context.level === NodeLevel.REL) {
            if (add_context.op !== null)
                throw new Error("Relation not followed by anything");
            return add_context;
        }
        let {node, op} = add_context;
        if (op !== null)
            throw new Error("Operator not followed by anything");
        contexts.shift();
        [context] = contexts;
        if (context.level !== NodeLevel.REL) {
            let rel_context: Context & {level: NodeLevel.REL} =
                {level: NodeLevel.REL, node, op: null};
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
            case Operation.EQ:
                rel_node = new ASTEq(context.node, node);
                break;
        }
        let rel_context: Context & {level: NodeLevel.REL} =
            {level: NodeLevel.REL, node: rel_node, op: null};
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
                if (rel_context.level === NodeLevel.GROUP)
                    throw new Error("Relation not preceded by anything")
                rel_context.op = Operation.GEQ;
                continue iterate_tokens;
            }
            case "<=": {
                let rel_context = reduce_to_rel(context);
                if (rel_context.level === NodeLevel.GROUP)
                    throw new Error("Relation not preceded by anything")
                rel_context.op = Operation.LEQ;
                continue iterate_tokens;
            }
            case "==": {
                let rel_context = reduce_to_rel(context);
                if (rel_context.level === NodeLevel.GROUP)
                    throw new Error("Relation not preceded by anything")
                rel_context.op = Operation.EQ;
                continue iterate_tokens;
            }
            case "+" : {
                if (
                    context.level === NodeLevel.GROUP ||
                    context.level === NodeLevel.REL ||
                    context.op !== null
                ) {
                    contexts.unshift({
                        level: NodeLevel.UNARY,
                        node: null,
                        op: Operation.ADD,
                    });
                    continue iterate_tokens;
                }
                let add_context = reduce_to_add(context);
                if (
                    add_context.level === NodeLevel.GROUP ||
                    add_context.level === NodeLevel.REL )
                    throw new Error("Operation not preceded by anything")
                add_context.op = Operation.ADD;
                continue iterate_tokens;
            }
            case "-" : {
                if (
                    context.level === NodeLevel.GROUP ||
                    context.level === NodeLevel.REL ||
                    context.op !== null
                ) {
                    contexts.unshift({
                        level: NodeLevel.UNARY,
                        node: null,
                        op: Operation.SUB,
                    });
                    continue iterate_tokens;
                }
                let add_context = reduce_to_add(context);
                if (
                    add_context.level === NodeLevel.GROUP ||
                    add_context.level === NodeLevel.REL )
                    throw new Error("Operation not preceded by anything")
                add_context.op = Operation.SUB;
                continue iterate_tokens;
            }
            case "*" : {
                let mul_context = reduce_to_mul(context);
                if (
                    mul_context.level === NodeLevel.GROUP ||
                    mul_context.level === NodeLevel.REL ||
                    mul_context.level === NodeLevel.ADD )
                    throw new Error("Operation not preceded by anything")
                mul_context.op = Operation.MUL;
                continue iterate_tokens;
            }
            case "/" : {
                let mul_context = reduce_to_mul(context);
                if (
                    mul_context.level === NodeLevel.GROUP ||
                    mul_context.level === NodeLevel.REL ||
                    mul_context.level === NodeLevel.ADD )
                    throw new Error("Operation not preceded by anything")
                mul_context.op = Operation.DIV;
                continue iterate_tokens;
            }
            case "(" : switch (context.level) {
                case NodeLevel.GROUP:
                    contexts.unshift({ level: NodeLevel.GROUP,
                        node: null, op: null });
                    continue iterate_tokens;
                case NodeLevel.REL:
                case NodeLevel.ADD:
                case NodeLevel.MUL:
                case NodeLevel.UNARY:
                    if (context.op === null)
                        throw new Error(
                            "Value is not preceded by operation" )
                    contexts.unshift({ level: NodeLevel.GROUP,
                        node: null, op: null });
                    continue iterate_tokens;
                default:
                    throw new Error("unreachable");
            }
            case ")" : {
                let rel_context = reduce_to_rel(context);
                if (rel_context.level === NodeLevel.GROUP)
                    throw new Error("Empty parentheses pair");
                contexts.shift();
                let opening_context = contexts.shift();
                if (opening_context === undefined || contexts.length === 0)
                    throw new Error("Unmatched closing parenthesis");
                if (opening_context.level !== NodeLevel.GROUP)
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
                case NodeLevel.GROUP:
                    contexts.unshift({ level: NodeLevel.MUL,
                        node: value, op: null });
                    continue iterate_tokens;
                case NodeLevel.REL:
                case NodeLevel.ADD:
                    if (context.op === null)
                        throw new Error(
                            "Value is not preceded by operation" )
                    contexts.unshift({ level: NodeLevel.MUL,
                        node: value, op: null });
                    continue iterate_tokens;
                case NodeLevel.MUL: switch(context.op) {
                    case null:
                        throw new Error(
                            "Value is not preceded by operation" )
                    case Operation.MUL:
                        contexts.shift();
                        contexts.unshift({ level: NodeLevel.MUL, op: null,
                            node: new ASTMul(context.node, value) });
                        continue iterate_tokens;
                    case Operation.DIV:
                        contexts.shift();
                        contexts.unshift({ level: NodeLevel.MUL, op: null,
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
    if (rel_context.level === NodeLevel.GROUP)
        throw new Error("Empty expression");
    return rel_context.node;
}

export class AffineFunction {
    linear: {[name: string]: number}
    constant: number;

    constructor(
        linear: {[name: string]: number},
        constant: number,
    ) {
        this.constant = constant;
        this.linear = linear;
    }

    negate(): AffineFunction {
        return this.scale(-1);
    }
    plus(other: AffineFunction): AffineFunction {
        let constant = this.constant + other.constant;
        let linear = Object.assign({}, this.linear);
        for (let [name, other_value] of Object.entries(other.linear)) {
            let value = linear[name];
            if (value !== undefined)
                other_value += value;
            linear[name] = other_value;
        }
        return new AffineFunction(linear, constant);
    }
    scale(scale: number): AffineFunction {
        return new AffineFunction(
            Object.fromEntries(Object.entries(this.linear).map(
                ([name, value]) => [name, scale * value] )),
            scale * this.constant,
        );
    }

    as_constant(): number | null {
        if (Object.values(this.linear).length == 0) {
            return this.constant;
        }
        return null;
    }

    static from_ast(ast: ASTExpression): AffineFunction {
        if (ast[EXPR_TYPE] === "constant") {
            return new AffineFunction({}, ast.value);
        }
        if (ast[EXPR_TYPE] === "variable") {
            return new AffineFunction({[ast.name]: 1}, 0);
        }
        if (ast[EXPR_TYPE] === "unary") {
            if (ast[OPERATION] !== Operation.SUB) {
                throw new Error("Unknown AST node type");
            }
            return this.from_ast(ast.value).negate();
        }
        if (ast[EXPR_TYPE] !== "binary") {
            throw new Error("Unknown AST node type");
        }
        if (ast[OPERATION] === Operation.ADD) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right) );
        }
        if (ast[OPERATION] === Operation.SUB) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right).negate() );
        }
        if (ast[OPERATION] === Operation.MUL) {
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
        if (ast[OPERATION] === Operation.DIV) {
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
        throw new Error("Unknown AST node type");
    }

    static from_tokens(tokens: Tokens): AffineFunction {
        let ast = parse_tokens(tokens);
        if (ast[NODE_TYPE] !== "expression")
            throw new Error("Can only parse an expression");
        return this.from_ast(ast);
    }

    *get_variables(): IterableIterator<string> {
        yield* Object.keys(this.linear);
    }

    as_covector(n: number, indices: {[name: string]: number}):
        Sparse.CoVector
    {
        let covector = Sparse.CoVector.zero(n + 1);
        if (Math.abs(this.constant) > 0)
            covector.set_value(n, this.constant);
        for (let [name, value] of Object.entries(this.linear)) {
            covector.set_value(indices[name], value);
        }
        return covector;
    }
}

export class AffineRelation {
    left: AffineFunction;
    relation: "eq" | "geq";

    constructor(
        left: AffineFunction,
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
        left: AffineFunction,
        relation: "eq" | "geq" | "leq",
        right: AffineFunction,
    ): AffineRelation {
        return new AffineRelation(left.plus(right.negate()), relation);
    }

    static from_ast(ast: ASTRelation): AffineRelation {
        if (ast[OPERATION] === Operation.GEQ) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "geq");
        }
        if (ast[OPERATION] === Operation.LEQ) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "leq");
        }
        if (ast[OPERATION] === Operation.EQ) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "eq");
        }
        throw new Error("Unknown AST node type");
    }

    static from_tokens(tokens: Tokens): AffineRelation {
        let ast = parse_tokens(tokens)
        if (!(ast[NODE_TYPE] === "relation"))
            throw new Error("Can only parse a relation");
        return this.from_ast(ast);
    }

    *get_variables(): IterableIterator<string> {
        yield* this.left.get_variables();
    }

    as_covector(n: number, indices: {[name: string]: number}):
        [Sparse.CoVector, "eq" | "geq"]
    {
        return [
            this.left.as_covector(n, indices),
            this.relation
        ];
    }
}

export class QuadraticFunction extends AffineFunction{
    quadratic: {[name1: string]: {[name2: string]: number}};

    constructor(
        quadratic: {[name1: string]: {[name2: string]: number}},
        linear: {[name: string]: number},
        constant: number,
    ) {
        super(linear, constant);
        this.quadratic = quadratic;
    }

    negate(): QuadraticFunction {
        return this.scale(-1);
    }
    plus(other: QuadraticFunction): QuadraticFunction {
        let sublinear = super.plus(other);
        let quadratic: {[name1: string]: {[name2: string]: number}}
            = Object.fromEntries(Object.entries(this.quadratic).map(
                ([name, values]) =>
                    [name, Object.assign({}, values)] ));
        for (let [name1, other_values] of Object.entries(other.quadratic)) {
            let values = quadratic[name1];
            if (values === undefined) {
                quadratic[name1] = values = {};
            }
            for (let [name2, other_value] of Object.entries(other_values)) {
                let value = values[name2];
                if (value !== undefined)
                    other_value += value;
                values[name2] = other_value;
            }
        }
        return new QuadraticFunction( quadratic,
            sublinear.linear, sublinear.constant );
    }
    scale(scale: number): QuadraticFunction {
        let sublinear = super.scale(scale);
        return new QuadraticFunction(
            Object.fromEntries(Object.entries(this.quadratic).map(
                (name1, values) => [ name1,
                    Object.fromEntries(Object.entries(values).map(
                        ([name2, value]) => [name2, scale * value] )),
                ] )),
            sublinear.linear,
            sublinear.constant,
        );
    }

    static product(left: AffineFunction, right: AffineFunction):
        QuadraticFunction
    {
        let constant = left.constant * right.constant;
        let linear: {[name: string]: number} = {};
        let add_linear =
            (l: {[name: string]: number}, c: number) => {
                if (Math.abs(c) <= EPSILON)
                    return;
                for (let [name, added_value] of Object.entries(l)) {
                    added_value *= c;
                    let value = linear[name];
                    if (value !== undefined)
                        added_value += value;
                    linear[name] = added_value;
                }
            };
        add_linear(right.linear, left.constant);
        add_linear(left.linear, right.constant);
        let quadratic: {[name1: string]: {[name2: string]: number}} = {};
        for (let [namex, valuex] of Object.entries(left.linear)) {
            for (let [namey, valuey] of Object.entries(right.linear)) {
                let [name1, name2] = (namex <= namey) ?
                    [namex, namey] : [namey, namex];
                let added_value = valuex * valuey;
                let values = quadratic[name1];
                if (values === undefined) {
                    quadratic[name1] = values = {};
                }
                let value = values[name2];
                if (value !== undefined)
                    added_value += value;
                values[name2] = added_value;
            }
        }
        return new QuadraticFunction(quadratic, linear, constant);
    }

    as_linear(): AffineFunction | null {
        if (Object.values(this.quadratic).length == 0) {
            return new AffineFunction(this.linear, this.constant);
        }
        return null;
    }

    as_constant(): number | null {
        let linear = this.as_linear();
        if (linear === null)
            return null;
        return linear.as_constant();
    }

    static from_ast(ast: ASTExpression): QuadraticFunction {
        if (ast[EXPR_TYPE] === "constant") {
            return new QuadraticFunction({}, {}, ast.value);
        }
        if (ast[EXPR_TYPE] === "variable") {
            return new QuadraticFunction({}, {[ast.name]: 1}, 0);
        }
        if (ast[EXPR_TYPE] === "unary") {
            if (ast[OPERATION] !== Operation.SUB) {
                throw new Error("Unknown AST node type");
            }
            return this.from_ast(ast.value).negate();
        }
        if (ast[EXPR_TYPE] !== "binary") {
            throw new Error("Unknown AST node type");
        }
        if (ast[OPERATION] === Operation.ADD) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right) );
        }
        if (ast[OPERATION] === Operation.SUB) {
            return this.from_ast(ast.left).plus(
                this.from_ast(ast.right).negate() );
        }
        if (ast[OPERATION] === Operation.MUL) {
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
            let
                left_linear = left.as_linear(),
                right_linear = right.as_linear();
            if (left_linear !== null && right_linear !== null) {
                return this.product(left_linear, right_linear);
            }
            throw new Error("Cannot build expression of degree > 2");
        }
        if (ast[OPERATION] === Operation.DIV) {
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
        throw new Error("Unknown AST node type");
    }

    static from_tokens(tokens: Tokens): QuadraticFunction {
        let ast = parse_tokens(tokens);
        if (ast[NODE_TYPE] !== "expression")
            throw new Error("Can only parse an expression");
        return this.from_ast(ast);
    }

    *get_variables(): IterableIterator<string> {
        yield* Object.keys(this.linear);
        for (let [name, values] of Object.entries(this.quadratic)) {
            yield name;
            yield *Object.keys(values);
        }
    }

    as_covector(n: number, indices: {[name: string]: number}):
        Sparse.CoVector
    {
        throw new Error("not implemented");
    }

    as_matrix(n: number, indices: {[name: string]: number}):
        Sparse.QuadraticMatrix
    {
        let matrix = Sparse.QuadraticMatrix.zero(n + 1);
        if (Math.abs(this.constant) > 0)
            matrix.set_value(n, n, this.constant);
        for (let [name, value] of Object.entries(this.linear)) {
            let index = indices[name];
            if (index >= n)
                throw new Sparse.DimensionError();
            matrix.set_value(n, index, value/2);
            matrix.set_value(index, n, value/2);
        }
        for (let [name1, values] of Object.entries(this.quadratic)) {
            let index1 = indices[name1];
            if (index1 >= n)
                throw new Sparse.DimensionError();
            for (let [name2, value] of Object.entries(values)) {
                let index2 = indices[name2];
                if (index2 >= n)
                    throw new Sparse.DimensionError();
                if (index1 === index2) {
                    matrix.add_value(index1, index1, value);
                } else {
                    matrix.add_value(index1, index2, value/2);
                    matrix.add_value(index2, index1, value/2);
                }
            }
        }
        return matrix;
    }

}

export class MatrixBuilder {
    variables: Set<string>;
    variable_count: number = 0;
    indices: {[name: string]: number} | null;

    constructor() {
        this.variables = new Set<string>();
        this.indices = null;
    }

    take_variables(
        object: AffineFunction | AffineRelation | QuadraticFunction,
    ): void {
        for (let name of object.get_variables()) {
            if (this.variables.has(name))
                continue;
            this.variables.add(name);
        }
    }

    set_as_complete(): CompleteMatrixBuilder {
        let index = 0;
        let indices = Object.fromEntries(
            function*(variables: Set<string>): Iterable<[string, number]> {
                for (let name of variables) {
                    yield [name, index++];
                }
            }(this.variables));
        this.variable_count = index;
        return Object.assign(this, {indices});
    }

    make_covector( this: CompleteMatrixBuilder,
        object: AffineFunction,
    ): Sparse.CoVector {
        return object.as_covector(this.variable_count, this.indices);
    }

    make_constraint( this: CompleteMatrixBuilder,
        object: AffineRelation
    ): [Sparse.CoVector, "eq" | "geq"] {
        return object.as_covector(this.variable_count, this.indices);
    }

    make_relation_matrices( this: CompleteMatrixBuilder,
        objects: AffineRelation[],
    ): {eq: Sparse.Matrix, geq: Sparse.Matrix} {
        let rows: Array<[Sparse.CoVector, "eq" | "geq"]> =
            objects.map( object =>
                object.as_covector(this.variable_count, this.indices) );
        return {
            eq: new Sparse.Matrix( this.variable_count + 1,
                rows
                    .filter(([, type]) => type === "eq")
                    .map(([covector, ]) => covector) ),
            geq: new Sparse.Matrix( this.variable_count + 1,
                rows
                    .filter(([, type]) => type === "geq")
                    .map(([covector, ]) => covector) ),
        };
    }

    make_quadratic_matrix( this: CompleteMatrixBuilder,
        object: QuadraticFunction,
    ): Sparse.QuadraticMatrix {
        return object.as_matrix(this.variable_count, this.indices);
    }

    unmake_vector( this: CompleteMatrixBuilder,
        vector: Sparse.Vector,
    ): {[name: string]: number} {
        return Object.fromEntries(Object.entries(this.indices).map(
            ([name, index]: [string, number]) =>
                [name, vector.get_value(index)] ));
    }

}

type CompleteMatrixBuilder =
    MatrixBuilder & {indices: {[name: string]: number}};
export type MaybeMatrixBuilder = (
    MatrixBuilder & {indices: null} |
    CompleteMatrixBuilder
);

}

}
