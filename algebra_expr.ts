namespace Algebra {

export namespace Expression {

export type Token = string | number;
export type Tokens = Array<Token>;

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

function parse_tokens(tokens: Tokens): ASTNode {
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
                rel_node = new ASTEq(context.node, node);
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

const EXPR_TYPE = Symbol();

type ASTNode = ASTExpression | ASTRelation;

type ASTExpression = (
    ASTConstant | ASTVariable | ASTUnary | ASTBinary
) & {[EXPR_TYPE]: "expression"};

class ASTExpressionClass {
    get [EXPR_TYPE](): "expression" {
        return "expression";
    }
};

class ASTConstant extends ASTExpressionClass {
    value: number;
    constructor(value: number) {
        super();
        this.value = value;
    }
}

class ASTVariable extends ASTExpressionClass {
    name: string;
    constructor(name: string) {
        super();
        this.name = name;
    }
}

type ASTUnary = ASTNeg;
class ASTUnaryBase extends ASTExpressionClass {
    value: ASTExpression;
    constructor(value: ASTNode) {
        super();
        if (!(value instanceof ASTExpressionClass))
            throw new Error("Unary operator applied to non-expression.");
        this.value = value;
    }
}

class ASTNeg extends ASTUnaryBase {}

type ASTBinary = ASTAdd | ASTSub | ASTMul | ASTDiv;
class ASTBinaryBase extends ASTExpressionClass {
    left: ASTExpression;
    right: ASTExpression;
    constructor(left: ASTNode, right: ASTNode) {
        super();
        if (
            !(left instanceof ASTExpressionClass) ||
            !(right instanceof ASTExpressionClass)
        )
            throw new Error("Binary operator applied to non-expressions.");
        this.left = left;
        this.right = right;
    }
}

class ASTAdd extends ASTBinaryBase {};
class ASTSub extends ASTBinaryBase {};

class ASTMul extends ASTBinaryBase {};
class ASTDiv extends ASTBinaryBase {};

type ASTRelation = (
    ASTGEq | ASTLEq | ASTEq
) & {[EXPR_TYPE]: "relation"};

class ASTRelationClass {
    get [EXPR_TYPE](): "relation" {
        return "relation";
    }

    left: ASTExpression;
    right: ASTExpression;
    constructor(left: ASTNode, right: ASTNode) {
        if (
            !(left instanceof ASTExpressionClass) ||
            !(right instanceof ASTExpressionClass)
        )
            throw new Error("Relation applied to non-expressions.");
        this.left = left;
        this.right = right;
    }
}

class ASTGEq extends ASTRelationClass {};
class ASTLEq extends ASTRelationClass {};
class ASTEq extends ASTRelationClass {};

export class AffineFunction {
    coefficients: {[name: string]: number}
    constant: number;

    constructor(
        coefficients: {[name: string]: number},
        constant: number
    ) {
        this.coefficients = coefficients;
        this.constant = constant;
    }

    negate(): AffineFunction {
        return this.scale(-1);
    }
    plus(other: AffineFunction): AffineFunction {
        let coefficients = Object.fromEntries(
            Object.entries(this.coefficients) );
        for (let [name, value] of Object.entries(other.coefficients)) {
            coefficients[name] = (coefficients[name] || 0) + value;
        }
        return new AffineFunction( coefficients,
            this.constant + other.constant );
    }
    scale(value: number): AffineFunction {
        return new AffineFunction(
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

    static from_ast(ast: ASTExpression): AffineFunction {
        if (ast instanceof ASTConstant) {
            return new AffineFunction({}, ast.value);
        }
        if (ast instanceof ASTVariable) {
            return new AffineFunction({[ast.name]: 1}, 0);
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
        throw new Error("Unrecognized AST expression node");
    }

    static from_tokens(tokens: Tokens): AffineFunction {
        let ast = parse_tokens(tokens);
        if (!(ast instanceof ASTExpressionClass))
            throw new Error("Can only parse an expression");
        return this.from_ast(ast);
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

    static from_ast(ast: ASTRelationClass): AffineRelation {
        if (ast instanceof ASTGEq) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "geq");
        }
        if (ast instanceof ASTLEq) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "leq");
        }
        if (ast instanceof ASTEq) {
            let
                left = AffineFunction.from_ast(ast.left),
                right = AffineFunction.from_ast(ast.right);
            return new AffineRelation(left.plus(right.negate()), "eq");
        }
        throw new Error("Unrecognized AST node");
    }

    static from_tokens(tokens: Tokens): AffineRelation {
        let ast = parse_tokens(tokens)
        if (!(ast instanceof ASTRelationClass))
            throw new Error("Can only parse a relation");
        return this.from_ast(ast);
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

}

}