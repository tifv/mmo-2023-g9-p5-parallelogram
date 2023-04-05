namespace Algebra {

/** n is number of rows, m is number of columns */
type Matrix<V> = Array<V> & {n: number, m: number};
/** n is number of elements/rows */
type Vector<V> = Array<V> & {n: number, m: 1};
/** m is number of elements/columns */
type CoVector<V> = Array<V> & {n: 1, m: number};

function new_matrix<V>(n: number, m: 1, fill?: V): Vector<V>
function new_matrix<V>(n: 1, m: number, fill?: V): CoVector<V>
function new_matrix<V>(n: number, m: number, fill?: V): Matrix<V>
function new_matrix<V>(n: number, m: number, fill?: V): Matrix<V> {
    let matrix = Object.assign(new Array<V>(n*m), {n, m});
    if (fill !== undefined) {
        matrix.fill(fill);
    }
    return matrix;
}

function copy_matrix<V>(matrix: Vector<V>): Vector<V>
function copy_matrix<V>(matrix: CoVector<V>): CoVector<V>
function copy_matrix<V>(matrix: Matrix<V>): Matrix<V>
function copy_matrix<V>(matrix: Matrix<V>): Matrix<V> {
    const {n, m} = matrix;
    let copy = Object.assign(new Array<V>(), {n, m});
    copy.push(...matrix);
    return copy;
}

function transpose<V>(matrix: Vector<V>): CoVector<V>
function transpose<V>(matrix: CoVector<V>): Vector<V>
function transpose<V>(matrix: Matrix<V>): Matrix<V>
function transpose<V>(matrix: Matrix<V>): Matrix<V> {
    const {n, m} = matrix;
    if (matrix.n == 1) {
        return Object.assign(
            Array.from(matrix), {n: m, m: 1} );
    }
    if (matrix.m == 1) {
        return Object.assign(
            Array.from(matrix), {n: 1, m: n} );
    }
    let transposed = new_matrix<V>(m, n);
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < m; ++j) {
            transposed[j*n+j] = matrix[i*m+j];
        }
    }
    return transposed;
}

function apply(covector: CoVector<number>, vector: Vector<number>): number {
    if (covector.m != vector.n) {
        throw new Error("Vector sizes do not match");
    }
    return covector.reduce((sum, y, i) => sum + y * vector[i], 0);
}

type Constraint = {
    covector: CoVector<number>,
    type: "eq",
    value: number,
} | {
    covector: CoVector<number>,
    type: "geq",
    value: number,
};

class GaussianReductor {
    old_n: number;
    new_n: number;
    consistent: boolean;
    private _constraints: Array<Constraint&{type: "eq"}>;
    private _leading_indices: Vector< number | null >;
    /** Either [new_index, null] or [null, eliminator_index] */
    private _index_map: CoVector< [number,null] | [null,number] >;
    private _rev_index_map: CoVector<number>;

    constructor(
        n: number,
        constraints: Array<Constraint&{type: "eq"}>
    ) {
        this.old_n = n;
        let consistent = true;
        let eliminated_by = new_matrix<number|null>(1, n, null);
        let eliminated_count = 0;
        let leading_indices =
            new_matrix<number|null>(constraints.length, 1, null);
        for (let [index, constraint] of constraints.entries()) {
            let leading_index: number | null = null; {
                let max_value = 0;
                for (let [j, value] of constraint.covector.entries()) {
                    if (eliminated_by[j] !== null)
                        continue;
                    value = Math.abs(value);
                    if (value < max_value + EPSILON)
                        continue;
                    leading_index = j;
                    max_value = value;
                }
            }
            if (leading_index === null) {
                if (Math.abs(constraint.value) > EPSILON) {
                    consistent = false;
                }
                continue;
            }
            leading_indices[index] = leading_index;
            GaussianReductor.scale( constraint,
                1/constraint.covector[leading_index] );
            constraint.covector[leading_index] = 1;
            for (let [i, other_constraint] of constraints.entries()) {
                if (i == index)
                    continue;
                GaussianReductor._eliminate( other_constraint,
                    leading_index, constraint );
            }
            eliminated_by[leading_index] = index;
            eliminated_count += 1;
        }
        let new_m = n - eliminated_count;
        let index_map = new_matrix<[number,null]|[null,number]>(1, n);
        let rev_index_map = new_matrix<number>(1, new_m);
        let image_index = 0;
        for (let i = 0; i < n; ++i) {
            let eliminated = eliminated_by[i];
            if (eliminated !== null) {
                index_map[i] = [null,eliminated];
                continue;
            }
            index_map[i] = [image_index,null];
            rev_index_map[image_index] = i;
            image_index += 1;
        }
        this.new_n = new_m;
        this.consistent = consistent;
        this._constraints = constraints;
        this._leading_indices = leading_indices;
        this._rev_index_map = rev_index_map;
        this._index_map = index_map;
    }
    static scale(constraint: Constraint, scale: number) {
        let covector = constraint.covector, {m} = covector;
        for (let i = 0; i < m; ++i) {
            covector[i] *= scale;
        }
        constraint.value *= scale;
    }
    static _eliminate(
        constraint: Constraint,
        index: number,
        eliminator: Constraint,
    ): number {
        let
            covector = constraint.covector, {m} = covector,
            eliminator_cov = eliminator.covector;
        let ratio = covector[index] / eliminator_cov[index];
        if (Math.abs(ratio) < EPSILON) {
            covector[index] = 0;
            return 0;
        }
        for (let i = 0; i < m; ++i) {
            let eliminator_element = eliminator_cov[i];
            if (eliminator_element === 0)
                continue;
            covector[i] -= eliminator_element * ratio;
        }
        covector[index] = 0;
        constraint.value -= eliminator.value * ratio;
        return ratio;
    }
    reduce_constraint( this: GaussianReductor & {consistent: true},
        constraint: Constraint,
    ): Constraint {
        if (!this.consistent)
            throw new Error(
                "Reduction impossible: constraints were inconsistent" );
        let m = this.new_n, rev_index_map = this._rev_index_map;
        let
            covector = new_matrix<number>(1, m, 0),
            type = constraint.type,
            value = constraint.value;
        for (let index = 0; index < this.old_n; ++index) {
            let preimage_element = constraint.covector[index];
            let index_status = this._index_map[index];
            if (index_status[1] === null) {
                covector[index_status[0]] += preimage_element;
                continue;
            }
            let
                eliminator = this._constraints[index_status[1]],
                eliminator_cov = eliminator.covector,
                ratio = preimage_element / eliminator_cov[index];
            value -= eliminator.value * ratio;
            for (let i = 0; i < m; ++i) {
                covector[i] -= eliminator_cov[rev_index_map[i]] * ratio;
            }
        }
        return {covector, type, value};
    }
    reduce_covector( this: GaussianReductor & {consistent: true},
        covector: CoVector<number>,
    ): CoVector<number> {
        if (!this.consistent)
            throw new Error(
                "Reduction impossible: constraints were inconsistent" );
        let m = this.new_n, rev_index_map = this._rev_index_map;
        let
            result = new_matrix<number>(1, m, 0);
        for (let index = 0; index < this.old_n; ++index) {
            let preimage_element = covector[index];
            let index_status = this._index_map[index];
            if (index_status[1] === null) {
                result[index_status[0]] += preimage_element;
                continue;
            }
            let
                eliminator = this._constraints[index_status[1]],
                eliminator_cov = eliminator.covector,
                ratio = preimage_element / eliminator_cov[index];
            for (let i = 0; i < m; ++i) {
                result[i] -= eliminator_cov[rev_index_map[i]] * ratio;
            }
        }
        return result;
    }
    project_vector(vector: Vector<number>): Vector<number> {
        let n = this.new_n, rev_index_map = this._rev_index_map;
        let result = new_matrix<number>(n, 1);
        for (let i = 0; i < n; ++i) {
            result[i] = vector[rev_index_map[i]];
        }
        return result;
    }
    recover_vector( this: GaussianReductor & {consistent: true},
        vector: Vector<number>,
    ): Vector<number> {
        if (!this.consistent)
            throw new Error(
                "Recover impossible: constraints were inconsistent" );
        let n = this.new_n, rev_index_map = this._rev_index_map;
        let result = new_matrix<number>(this.old_n, 1);
        for (let index = 0; index < this.old_n; ++index) {
            let index_status = this._index_map[index]
            if (index_status[1] === null) {
                result[index] = vector[index_status[0]];
                continue;
            }
            let
                eliminator = this._constraints[index_status[1]],
                eliminator_cov = eliminator.covector;
            let value = eliminator.value;
            for (let i = 0; i < n; ++i) {
                value -= vector[i] * eliminator_cov[rev_index_map[i]];
            }
            result[index] = value / eliminator_cov[index];
        }
        return result;
    }
}

type MaybeReductor = (
    GaussianReductor & {consistent: true} |
    GaussianReductor & {consistent: false}
);

function evaluate_constraint(constraint: Constraint, point: Vector<number>): {
    satisfied: boolean, exact: boolean, difference: number }
{
    let {covector, type, value} = constraint;
    let difference = apply(covector, point) - value;
    let exact = Math.abs(difference) < EPSILON;
    return {
        satisfied: (type === "eq") ? exact : difference > -EPSILON,
        exact,
        difference,
    };
}

type LinearObjective = CoVector<number>;
type LinearTarget = number | "min"
// type QuadraticObjective = {
//     quadratic: Matrix<number>,
//     linear: CoVector<number>,
// };

type Solution = Vector<number> & {error?: false};
type ErrorReport = {error: true, description: string};


type UnitColumn = {row: number, values: null};
type ValueColumn = {row: null, values: Vector<number>, objective: number};
type LPState = {
    k0: number,
    point: Vector<number>,
    row_index_map: Array<
        {tableau_col: number, residual_col: null  } |
        {tableau_col: null,   residual_col: number}
    >;
    tableau: Array<UnitColumn|ValueColumn>,
    residuals: Array<UnitColumn|ValueColumn>,
    differences: Array<number|null>,
    result: false | Solution | ErrorReport,
}

export class LPProblem {
    n: number;
    k: number;
    objective: LinearObjective;
    target: LinearTarget;
    constraints: Array<Constraint>;
    private _result: Vector<number>;
    private _result_feasible: boolean;

    constructor(
        {
            objective,
            target,
            constraints,
            initial,
        }: {
            objective: LinearObjective,
            target: LinearTarget,
            constraints: Array<Constraint>,
            initial?: { point: Vector<number>, feasible: boolean},
        }
    ) {
        this.n = objective.m;
        this.objective = objective;
        this.target = target; // XXX not used yet
        this.k = constraints.length;
        this.constraints = constraints;
        if (initial === undefined) {
            this._result = new_matrix<number>(this.objective.m, 1, 0)
            this._result_feasible = false;
        } else {
            this._result = initial.point;
            this._result_feasible = initial.feasible;
        }
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
        let problem = new LPProblem({
            objective: objective_expr.as_covector(n, indices)[0],
            target,
            constraints: constraints_rels.map( constraint => 
                constraint.as_constraint(n, indices) ),
        });
        let solution = problem.solve();
        if (solution.error) {
            return solution;
        }
        if (solution.length != n) {
            throw new Error("unreachable");
        }
        let successful = solution;
        return {solution: Object.fromEntries(variables.map(
            (name, index) => [name, successful[index]] ))};
    }

    static find_feasible_point(
        n: number,
        constraints: Array<Constraint&{type: "geq"}>,
        initial?: Vector<number>,
    ): Solution | ErrorReport {
        let point = (initial !== undefined) ? initial :
            new_matrix<number>(n, 1, 0);
        let constraints_satisfied = constraints.map<[boolean,number]>(
            constraint => {
                let {satisfied, difference} =
                    evaluate_constraint(constraint, point);
                return [satisfied, difference];
            }
        );
        if (constraints_satisfied.every(([satisfied, ]) => satisfied)) {
            return point;
        }
        let feasibility_objective = new_matrix<number>(1, n + 1, 0);
        feasibility_objective[n] = +1;
        let feasibility_initial = copy_matrix(point);
        feasibility_initial.n = n + 1;
        feasibility_initial.push(Math.max(...constraints_satisfied.map(
            ([satisfied, difference]) => satisfied? 0 : -difference )));
        let feasibility_constraints = constraints.map( (constraint) => {
            let covector = copy_matrix(constraint.covector);
            covector.m = n + 1;
            covector.push(+1);
            return { covector,
                type: constraint.type, value: constraint.value };
        });
        feasibility_constraints.push({
            covector: feasibility_objective,
            type: "geq",
            value: 0
        });
        let feasibility_problem = new LPProblem({
            objective: feasibility_objective,
            target: 0,
            constraints: feasibility_constraints,
            initial: { point: feasibility_initial, feasible: true },
        });
        let feasibility_solution = feasibility_problem.solve();
        if (feasibility_solution?.error) {
            return feasibility_solution;
        }
        if (feasibility_solution[n] > EPSILON) {
            return {error: true, description: "Infeasible"};            
        }
        let feasible_point: Vector<number> =
            Object.assign( feasibility_solution.slice(0, n),
                {n, m: <1>1} );
        return feasible_point;
    }

    solve(): Solution | ErrorReport {
        let
            eq_constraints = new Array<Constraint&{type:"eq"}>(),
            geq_constraints = new Array<Constraint&{type:"geq"}>();
        for (let constraint of this.constraints) {
            if (constraint.type == "eq") {
                eq_constraints.push(constraint);
            } else {
                geq_constraints.push(constraint);
            }
        }
        if (eq_constraints.length != 0) {
            let maybe_reductor: MaybeReductor =
                new GaussianReductor(this.n, eq_constraints);
            if (!maybe_reductor.consistent)
                return { error: true,
                    description: "Equations are inconsistent" };
            let reductor = maybe_reductor;
            // let recover_result = (solution:) => {
            //     return reductor.recover_vector(solution);
            // }
            let reduced_objective: Constraint = reductor.reduce_constraint({
                covector: this.objective,
                type: "eq",
                value: this.target == "min" ? 0 : this.target });
            let result = new LPProblem({
                objective: reduced_objective.covector,
                target: this.target == "min" ? "min" : reduced_objective.value,
                constraints: geq_constraints.map( constraint =>
                    reductor.reduce_constraint(constraint) ),
                initial: {
                    point: reductor.project_vector(this._result),
                    feasible: this._result_feasible,
                }
            }).solve();
            if (result.error)
                return result;
            return reductor.recover_vector(result);
        }
        let feasible_point: Vector<number>;
        if (this._result_feasible) {
            feasible_point = this._result;
        } else {
            let feasibility_solution = LPProblem.find_feasible_point(
                this.n, geq_constraints, this._result );
            if (feasibility_solution.error)
                return feasibility_solution;
            feasible_point = feasibility_solution;
        }
        let state = this._build_initial_state(feasible_point);
        while (state.result === false) {
            state = this._step(state);
        }
        let result = state.result;
        if (result.error)
            return result;
        return result;
    }

    _build_initial_state(
        feasible_point: Vector<number>,
    ): LPState {
        const {n, k} = this;
        let k0: number = 0;
        let point = feasible_point;
        let row_index_map = new Array<{
            tableau_col: number, residual_col: null,
        }>();
        let tableau = new Array<UnitColumn>();
        for (let index = 0; index < k; ++index) {
            row_index_map[index] = {tableau_col: index, residual_col: null};
            tableau[index] = {row: index, values: null};
        }

        let residuals = new Array<{
            row: null, values: Vector<number>, objective: number,
        }>();
        for (let index = 0; index < n; ++index) {
            residuals.push({ row: null,
                values: Object.assign(new Array(k), {n: k, m: <1>1}),
                objective: this.objective[index] });
        }
        let differences = new Array<number>(k);
        this.constraints.forEach((constraint, index) => {
            if (constraint.type == "eq")
                throw new Error("unreachable");
            for (let [i, value] of constraint.covector.entries()) {
                residuals[i].values[index] = value;
            }
            let
                value = apply(constraint.covector, point),
                difference = value - constraint.value;
            differences[index] = difference;
            if (difference < -EPSILON) {
                throw new Error("unreachable");
            }
        });

        return {
            k0, point,
            row_index_map, tableau,
            residuals, differences,
            result: false,
        };
    }

    /** somewhat expensive check, debug-only */
    _validate_state(state: LPState): void {
        const {n, k} = this;
        for (let j = 0; j < k; ++j) {
            let constraint = this.constraints[j];
            if (constraint.type === "eq")
                throw new Error("unreachable");
            let difference: number; {
                let tableau_col = state.tableau[j];
                if (tableau_col.values !== null) {
                    difference = 0;
                } else {
                    let maybe_difference = state.differences[tableau_col.row];
                    if (maybe_difference === null)
                        throw new Error("unreachable");
                    difference = maybe_difference;
                }
            }
            let true_difference =
                apply(constraint.covector, state.point) - constraint.value;
            if (Math.abs(true_difference - difference) > EPSILON) {
                throw new Error("State is invalid");
            }
        }
    }

    _step(state: LPState): LPState {
        let leaving_item = this._find_leaving_index(state);
        if (leaving_item === null) {
            return Object.assign({}, state, {result: state.point});
        }
        let entering_index: number; {
            let entering_index_info:
                {index: number, error?: false} | ErrorReport;
            if (leaving_item?.variable !== undefined) {
                entering_index_info = this._find_entering_index(
                    state, leaving_item.variable );
            } else {
                entering_index_info = this._find_replacing_index(
                    state, leaving_item.constraint );
            }
            if (entering_index_info.error) {
                return Object.assign({}, state, {result: entering_index_info});
            }
            entering_index = entering_index_info.index;
        }
        if (leaving_item.variable !== undefined) {
            return this._pivot_entering(state, leaving_item, entering_index);
        } else {
            return this._pivot_replacing(state, leaving_item, entering_index);
        }
    }

    _find_leaving_index(state: LPState): (
        null |
        { variable: {index: number, residual_column: ValueColumn, sign: 1 | -1},
            constraint?: undefined,
            vector: Vector<number> } |
        { constraint: {index: number, tableau_column: ValueColumn},
            variable?: undefined,
            vector: Vector<number> }
    ) {
        const {n} = this;
        let max_residual: {index: number, column: ValueColumn} | null = null; {
            let max_value = 0;
            for (let [index, column] of state.residuals.entries()) {
                if (column.values === null)
                    continue;
                let value = Math.abs(column.objective);
                if (value < max_value + EPSILON)
                    continue;
                max_residual = {index, column};
                max_value = value;
            }
        }
        if (max_residual !== null) {
            let leaving_index = max_residual.index;
            let value: 1 | -1 = max_residual.column.objective > 0 ? -1 : 1;
            let leaving_vector: Vector<number> = Object.assign(
                new Array<number>(), {n, m: <1>1} );
            let residual_values = max_residual.column.values;
            for (let i = 0; i < n; ++i) {
                let column = state.residuals[i];
                if (column.row === null) {
                    leaving_vector[i] = 0;
                } else {
                    leaving_vector[i] = -value * residual_values[column.row];
                }
            }
            leaving_vector[leaving_index] = value;
            return {
                variable: { index: leaving_index, sign: value,
                    residual_column: max_residual.column },
                vector: leaving_vector,
            };
        }
        for (let [index, column] of state.tableau.entries()) {
            if (column.values === null || column.objective < EPSILON)
                continue;
            let leaving_vector: Vector<number> = Object.assign(
                new Array<number>(), {n, m: <1>1} );
            let leaving_values = column.values;
            for (let i = 0; i < n; ++i) {
                let i_column = state.residuals[i];
                if (i_column.row === null) {
                    leaving_vector[i] = 0;
                } else {
                    leaving_vector[i] = leaving_values[i_column.row];
                }
            }
            return { constraint: {index: index, tableau_column: column},
                vector: leaving_vector };
        }
        return null;
    }

    _find_entering_index( state: LPState,
        leaving_variable: { index: number, sign: 1 | -1,
            residual_column: ValueColumn },
    ): {index: number, error?: false} | ErrorReport
    {
        let
            entering_index: null | number = null,
            min_ratio: null | number = null;
        let leaving_values = leaving_variable.residual_column.values;
        let differences = state.differences;
        for (let [index, index_info] of state.row_index_map.entries()) {
            if (index_info.tableau_col === null)
                continue;
            let
                leaving_value = leaving_values[index] * leaving_variable.sign,
                difference = differences[index] || 0;
            if (leaving_value > -EPSILON)
                continue;
            if (difference < EPSILON)
                return {index}; // Bland's rule, kinda
            let ratio = difference / -leaving_value;
            if (ratio < -EPSILON) {
                throw new Error("unreachable");
            }
            if (min_ratio === null || ratio < min_ratio) {
                min_ratio = ratio;
                entering_index = index;
            }
        }
        if (entering_index === null) {
            return {error: true, description: "Unbounded"};
        }
        return {index: entering_index};
    }

    _find_replacing_index( state: LPState,
        leaving_constraint: {index: number, tableau_column: ValueColumn},
    ): {index: number, error?: false} | ErrorReport {
        let
            entering_index: null | number = null,
            min_ratio: null | number = null;
        let leaving_values = leaving_constraint.tableau_column.values;
        let differences = state.differences;
        for (let [index, index_info] of state.row_index_map.entries()) {
            if (index_info.tableau_col === null)
                continue;
            let
                leaving_value = leaving_values[index],
                difference = differences[index] || 0;
            if (leaving_value < EPSILON)
                continue;
            if (difference < EPSILON)
                return {index}; // Bland's rule
            let ratio = difference / leaving_value;
            if (ratio < -EPSILON) {
                throw new Error("unreachable");
            }
            if (min_ratio === null || ratio < min_ratio) {
                min_ratio = ratio;
                entering_index = index;
            }
        }
        if (entering_index === null) {
            return {error: true, description: "Unbounded"};
        }
        return {index: entering_index};
    }

    _pivot_entering(
        {
            k0,
            row_index_map,
            tableau,
            residuals,
            differences,
            point,
        }: LPState,
        leaving_item: {
            variable: {index: number, residual_column: ValueColumn, sign: 1 | -1},
            vector: Vector<number> },
        entering_index: number,
    ): LPState {
        const {n, k} = this;
        point = copy_matrix(point);
        row_index_map = Array.from(row_index_map);
        tableau = tableau.map(column => {
            if (column.row !== null)
                return {row: column.row, values: null};
            return { row: null,
                values: copy_matrix(column.values),
                objective: column.objective };
        });
        residuals = residuals.map(column => {
            if (column.row !== null)
                return {row: column.row, values: null};
            return { row: null,
                values: copy_matrix(column.values),
                objective: column.objective };
        });
        differences = Array.from(differences);

        let leaving_variable = leaving_item.variable;
        let leaving_column = leaving_variable.residual_column;
        let leaving_values: Array<number> = leaving_column.values;
        let leaving_objective = leaving_column.objective;
        let leaving_index = leaving_variable.index;

        // shift the point
        let vector = leaving_item.vector;
        let entering_difference = differences[entering_index];
        if (entering_difference === null)
            throw new Error("unreachable");
        let vector_scale: number = -entering_difference / (
            vector[leaving_index] * leaving_values[entering_index] );
        if (vector_scale < -EPSILON)
            throw new Error("unreachable");
        for (let i = 0; i < n; ++i) {
            point[i] += vector_scale * vector[i];
        }
        let difference_scale = vector_scale * vector[leaving_index];
        for (let [index, difference] of differences.entries()) {
            if (difference === null)
                continue;
            difference += leaving_values[index] * difference_scale;
            if (difference < -EPSILON)
                throw new Error("unreachable");
            differences[index] = difference;
        }
        differences[entering_index] = null;

        { // reconstruct the entering column
            let entering_values: Vector<number> = new_matrix<number>(k, 1, 0);
            entering_values[entering_index] = 1;
            let entering_col =
                row_index_map[entering_index].tableau_col;
            if (entering_col === null)
                throw new Error("unreachable");
            tableau[entering_col] = {
                row: null,
                values: entering_values,
                objective: 0,
            };
        }

        { // scale entering row
            let leading_value = leaving_values[entering_index];
            for (let column of tableau) {
                if (column.values === null)
                    continue;
                column.values[entering_index] /= leading_value;
            }
            for (let column of residuals) {
                if (column.values === null)
                    continue;
                column.values[entering_index] /= leading_value;
            }
            leaving_values[entering_index] = 1;
        }

        // save leaving_values
        leaving_values = Array.from(leaving_values);
        // eliminate with the entering row
        for (let column of tableau) {
            if (column.values === null)
                continue;
            let values = column.values;
            let eliminator_element = values[entering_index];
            for (let i = 0; i < leaving_values.length; ++i) {
                if (i == entering_index)
                    continue;
                values[i] -= eliminator_element * leaving_values[i];
            }
            column.objective -= eliminator_element * leaving_objective;
        }
        for (let column of residuals) {
            if (column.values === null)
                continue;
            let values = column.values;
            let eliminator_element = values[entering_index];
            for (let i = 0; i < leaving_values.length; ++i) {
                if (i == entering_index)
                    continue;
                values[i] -= eliminator_element * leaving_values[i];
            }
            column.objective -= eliminator_element * leaving_objective;
        }

        row_index_map[entering_index] =
            {tableau_col: null, residual_col: leaving_index};
        residuals[leaving_index] = {row: entering_index, values: null};
        return {
            k0: k0 + 1,
            point,
            row_index_map,
            tableau,
            residuals,
            differences,
            result: false,
        };
    }

    _pivot_replacing(
        {
            k0,
            row_index_map,
            tableau,
            residuals,
            differences,
            point,
        }: LPState,
        leaving_item: {
            constraint: {index: number, tableau_column: ValueColumn},
            vector: Vector<number> },
        entering_index: number,
    ): LPState {
        const {n, k} = this;
        point = copy_matrix(point);
        row_index_map = Array.from(row_index_map);
        tableau = tableau.map(column => {
            if (column.row !== null)
                return {row: column.row, values: null};
            return { row: null,
                values: copy_matrix(column.values),
                objective: column.objective };
        });
        residuals = residuals.map(column => {
            if (column.row !== null)
                return {row: column.row, values: null};
            return { row: null,
                values: copy_matrix(column.values),
                objective: column.objective };
        });
        differences = Array.from(differences);

        let leaving_constraint = leaving_item.constraint;
        let leaving_column = leaving_constraint.tableau_column;
        let leaving_values: Array<number> = leaving_column.values;
        let leaving_objective = leaving_column.objective;
        let leaving_index = leaving_constraint.index;

        // shift the point
        let vector = leaving_item.vector;
        let entering_difference = differences[entering_index];
        if (entering_difference === null)
            throw new Error("unreachable");
        let vector_scale: number =
            entering_difference / leaving_values[entering_index];
        if (vector_scale < -EPSILON)
            throw new Error("unreachable");
        for (let i = 0; i < n; ++i) {
            point[i] += vector_scale * vector[i];
        }

        { // reconstruct the entering column
            let entering_values: Vector<number> = new_matrix<number>(k, 1, 0);
            entering_values[entering_index] = 1;
            let entering_col =
                row_index_map[entering_index].tableau_col;
            if (entering_col === null)
                throw new Error("unreachable");
            tableau[entering_col] = {
                row: null,
                values: entering_values,
                objective: 0,
            };
        }

        { // scale entering row
            let leading_value = leaving_values[entering_index];
            for (let column of tableau) {
                if (column.values === null)
                    continue;
                column.values[entering_index] /= leading_value;
            }
            for (let column of residuals) {
                if (column.values === null)
                    continue;
                column.values[entering_index] /= leading_value;
            }
            differences[entering_index] = entering_difference =
                entering_difference / leading_value;
            leaving_values[entering_index] = 1;
        }

        // save leaving_values
        leaving_values = Array.from(leaving_values);

        for (let column of tableau) {
            if (column.values === null)
                continue;
            let values = column.values;
            let eliminator_element = values[entering_index];
            for (let i = 0; i < leaving_values.length; ++i) {
                if (i == entering_index)
                    continue;
                values[i] -= eliminator_element * leaving_values[i];
            }
            column.objective -= eliminator_element * leaving_objective;
        }
        for (let column of residuals) {
            if (column.values === null)
                continue;
            let values = column.values;
            let eliminator_element = values[entering_index];
            for (let i = 0; i < leaving_values.length; ++i) {
                if (i == entering_index)
                    continue;
                values[i] -= eliminator_element * leaving_values[i];
            }
            column.objective -= eliminator_element * leaving_objective;
        }
        for (let i = 0; i < leaving_values.length; ++i) {
            if (i == entering_index)
                continue;
            let difference = differences[i];
            if (difference === null)
                continue;
            difference -= entering_difference * leaving_values[i];
            if (difference < -EPSILON)
                throw new Error("unreachable");
            differences[i] = difference;
        }

        row_index_map[entering_index] =
            {tableau_col: leaving_index, residual_col: null};
        tableau[leaving_index] = {row: entering_index, values: null};
        return {
            k0: k0 + 1,
            point,
            row_index_map,
            tableau,
            residuals,
            differences,
            result: false,
        };
    }

}

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

    as_covector(n: number, indices: {[name: string]: number}):
        [CoVector<number>, number]
    {
        let covector = new_matrix(1, n, 0);
        for (let [name, value] of Object.entries(this.coefficients)) {
            covector[indices[name]] = value;
        }
        return [covector, this.constant];
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

    as_constraint(n: number, indices: {[name: string]: number}): Constraint {
        let covector = new_matrix(1, n, 0);
        for (let [name, value] of Object.entries(this.left.coefficients)) {
            covector[indices[name]] = value;
        }
        return {
            covector: covector,
            type: this.relation,
            value: -this.left.constant,
        };
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
    let solution = Algebra.LPProblem.solve_from_tokens(
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
