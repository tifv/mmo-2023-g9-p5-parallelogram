namespace Algebra {

export namespace Solver {

import Vector = Sparse.Vector;
import CoVector = Sparse.CoVector;
import Matrix = Sparse.Matrix;
import QuadraticMatrix = Algebra.Sparse.QuadraticMatrix;

type Solution = Vector;
type ErrorReport = {error: true, description: string};

export class LPProblemSolver {
    n: number;
    k: number;
    private k0: number;
    private k1: number;
    private point: Vector;
    private row_indices: Array<(
        {tableau_col: number, variable_col?: undefined} |
        {variable_col: number, tableau_col?: undefined}
    )>;
    private col_indices: {
        constraint: Array<number|null>,
        variable: Array<number|null>,
    };
    private tableau: {
        constraint: Matrix,
        objective: CoVector,
    };
    private result: null | Solution | ErrorReport;

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
            let maybe_reductor: Sparse.MaybeConsistentReductor =
                new Sparse.GaussianReductor(constraints_eq);
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
        objective: Expression.Token[],
        target: "min",
        constraints: Array<Expression.Token[]>,
    ): {error?: false, solution: {[name: string]: number}} | ErrorReport {
        let objective_expr =
            Expression.AffineFunction.from_tokens(objective);
        let constraints_rels = constraints.map( tokens =>
            Expression.AffineRelation.from_tokens(tokens) );
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
            throw new Sparse.DimensionError();
        if (feasible_point.height !== n)
            throw new Sparse.DimensionError();

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

} // end namespace

} // end namespace


// replacement for js-lp-solver
var solver = {
Solve: function Solve(model: LPModel): { [s: string]: any; } {
    type Token = Algebra.Expression.Token;
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
    let solution = Algebra.Solver.LPProblemSolver.solve_from_tokens(
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
