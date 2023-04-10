namespace Algebra {

export namespace Solver {

import Vector = Sparse.Vector;
import CoVector = Sparse.CoVector;
import Matrix = Sparse.Matrix;
import QuadraticMatrix = Algebra.Sparse.QuadraticMatrix;

type Solution = Vector;
type ErrorReport = {error: true, description: string};

import Tokens = Algebra.Expression.Tokens;

export type LPProblemTokenized = {
    objective: Tokens,
    target: "min" | "max",
    constraints: Array<Tokens>,
}

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
        {
            objective: obj_tokens,
            target,
            constraints: cstr_tokens,
        } : LPProblemTokenized,
    ): {error?: false, solution: {[name: string]: number}} | ErrorReport {
        let objective_expr =
            Expression.AffineFunction.from_tokens(obj_tokens);
        let constraints_rels = cstr_tokens.map( tokens =>
            Expression.AffineRelation.from_tokens(tokens) );
        if (target === "max")
            objective_expr = objective_expr.negate();

        let matrix_builder: Expression.MaybeMatrixBuilder =
            new Expression.MatrixBuilder();
        matrix_builder.take_variables(objective_expr);
        constraints_rels.forEach(rel => matrix_builder.take_variables(rel));
        matrix_builder = matrix_builder.set_as_complete();
        const n = matrix_builder.variable_count;

        let {eq: constraints_eq, geq: constraints_geq} =
            matrix_builder.make_relation_matrices(constraints_rels);
        let solution = LPProblemSolver.solve({
            objective: matrix_builder.make_covector(objective_expr),
            constraints_eq, constraints_geq,
        });
        if (solution?.error) {
            return solution;
        }
        if (solution.height != n) {
            throw new Error("unreachable");
        }
        let successful = solution;
        return {solution: matrix_builder.unmake_vector(successful)};
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
                    tableau.add_value(i, i, 1);
                }
                return tableau;
            })(),
            objective: ((): CoVector => {
                let obj_residual = objective.relative_to(this.point);
                let objective_tableau = CoVector.zero(k + n + 1);
                objective_tableau.add_from( obj_residual,
                    {map: (index) => k + index},
                );
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
        if (!isFinite(vector_scale)) {
            this.result = {error: true, description: "Unbounded"};
            return;
        }

        if (vector_scale > EPSILON) {
            // shift the point
            for (let [index, value] of constraint_vector.iter_items()) {
                this.tableau.constraint.rows[index]
                    .add_value(k + n, vector_scale * -value);
            }
            let scaled_vector = Vector.from_items( n,
                entering_vector.map_items( (index, value) =>
                    [index, vector_scale * value] )
            );
            this.tableau.objective.shift(scaled_vector, {start_offset: k});
            this.point.add_from(scaled_vector);
        }

        if (entering === null) {
            throw new Error("unreachable");
        }
        let entering_index = entering.index;
        let entering_row = this.tableau.constraint.rows[entering_index];
        // XXX debug
        if (entering_row.some_nonzero(() => true, k + n)) {
            throw new Error("unreachable");
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
        this: LPProblemSolver | QPProblemSolver,
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

export type QPProblemTokenized = {
    objective: Tokens,
    target: "min",
    constraints: Array<Tokens>,
    check_defineteness?: boolean,
}

export class QPProblemSolver {
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
        objective: QuadraticMatrix,
    };
    result: null | Solution | ErrorReport;

    static solve(
        {
            objective,
            constraints_eq,
            constraints_geq,
            check_defineteness = false
        }: {
            objective: QuadraticMatrix,
            constraints_eq: Matrix | null,
            constraints_geq: Matrix,
            check_defineteness?: boolean
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
            objective = reductor.reduce_affine_quadratic(objective.copy());
        }
        let feasible_point =
            LPProblemSolver._find_feasible_point({ constraints:
                constraints_geq });
        if ("error" in feasible_point) {
            return feasible_point;
        }
        if (check_defineteness) {
            let definiteness = objective.get_definiteness();
            if (!definiteness.is_semipositive)
                return { error: true,
                    description: "Objective is not semi-positive" };
        }
        let solution = new QPProblemSolver({
            objective,
            constraints: constraints_geq,
            feasible_point,
        }).get_solution();
        if ("error" in solution)
            return solution;
        return reconstruct_solution(solution);
    }

    static solve_from_tokens(
        {
            objective: obj_tokens,
            target,
            constraints: cstr_tokens,
            check_defineteness = false,
        } : QPProblemTokenized,
    ): {error?: false, solution: {[name: string]: number}} | ErrorReport {
        let objective_expr =
            Expression.QuadraticFunction.from_tokens(obj_tokens);
        let constraints_rels = cstr_tokens.map( tokens =>
            Expression.AffineRelation.from_tokens(tokens) );
        if (target !== "min")
            throw new Error("Can only minimize quadratic objective");

        let matrix_builder: Expression.MaybeMatrixBuilder =
            new Expression.MatrixBuilder();
        matrix_builder.take_variables(objective_expr);
        constraints_rels.forEach(rel => matrix_builder.take_variables(rel));
        matrix_builder = matrix_builder.set_as_complete();
        const n = matrix_builder.variable_count;

        let {eq: constraints_eq, geq: constraints_geq} =
            matrix_builder.make_relation_matrices(constraints_rels);
        let solution = this.solve({
            objective: matrix_builder.make_quadratic_matrix(objective_expr),
            constraints_eq, constraints_geq,
            check_defineteness,
        });
        if (solution?.error) {
            return solution;
        }
        if (solution.height != n) {
            throw new Error("unreachable");
        }
        let successful = solution;
        return {solution: matrix_builder.unmake_vector(successful)};
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
                    tableau.add_value(i, i, 1);
                }
                return tableau;
            })(),
            objective: ((): QuadraticMatrix => {
                let obj_residual = objective.relative_to(this.point);
                let objective_tableau =
                    QuadraticMatrix.zero(k + n + 1);
                objective_tableau.add_from( obj_residual,
                    {map: (index) => k + index},
                    {map: (index) => k + index},
                );
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

    _step(): void {
        const {n, k} = this;
        let obj_tableau = this.tableau.objective;
        for ( let [index, value]
                of obj_tableau.rows[k + n].iter_items(k, k + n) ) {
            if (Math.abs(value) < EPSILON)
                continue;
            return this._step_enter();
        }
        for (let [index, value] of obj_tableau.rows[k + n].iter_items(0, k)) {
            if (value > -EPSILON)
                continue;
            return this._step_exit(index);
        }
        this.result = this.point;
    }

    _step_enter(): void {
        const {n, k} = this;
        let {entering_vector_base, entering_obj_limit} =
            this._find_entering_vector();
        let
            entering_vector: Vector = entering_vector_base.copy(),
            constraint_vector = Vector.zero(k),
            entering: {index: number, cstr_index: number} | null = null,
            vector_scale: number = entering_obj_limit;
        { // find entering index
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

        if (!isFinite(vector_scale)) {
            this.result = {error: true, description: "Unbounded"};
            return;
        }
        if (vector_scale > EPSILON) {
            // shift the point
            for (let [index, value] of constraint_vector.iter_items()) {
                this.tableau.constraint.rows[index]
                    .add_value(k + n, vector_scale * -value);
            }
            let scaled_vector = Vector.from_items( n,
                entering_vector.map_items( (index, value) =>
                    [index, vector_scale * value] )
            );
            this.tableau.objective.shift(scaled_vector, {start_offset: k});
            this.point.add_from(scaled_vector);
        }

        if (entering === null) {
            let linear_row = this.tableau.objective.rows[k + n];
            // XXX debug
            if (linear_row.some_nonzero(() => true, k, k + n)) {
                throw new Error("unreachable");
            }
            let items = Array.from(linear_row.iter_items(k, k + n));
            for (let [index, ] of items) {
                linear_row.set_value(index, null);
                this.tableau.objective.set_value(index, k + n, null);
            }
            return;
        }
        let entering_index = entering.index;
        let entering_row = this.tableau.constraint.rows[entering_index];
        // XXX debug
        if (entering_row.some_nonzero(() => true, k + n)) {
            throw new Error("unreachable");
        }
        entering_row.set_value(k + n, null);

        let leading_search = entering_row.scale_extreme(k, k + n, {
            direction: "abs", value_sign: "any", preserve_sign: false,
        });
        if (leading_search === null) {
            throw new Error("unreachable");
        }
        let leading_var_index = leading_search.index - k;
        this.tableau.constraint.eliminate_with({
            row: entering_row, index: k + leading_var_index, value: 1,
            is_within: true });
        this.tableau.objective.eliminate_with({
            row: entering_row, index: k + leading_var_index, value: 1 });

        this.col_indices.constraint[entering.cstr_index] = null;
        this.col_indices.variable[leading_var_index] = entering_index;
        this.row_indices[entering_index] = {variable_col: leading_var_index};

        this.k0 += 1;
        this.k1 -= 1;
    }

    _find_entering_vector(): {
        entering_vector_base: Vector,
        entering_obj_limit: number,
    } {
        const {n, k} = this;
        let objective_minimum_finder = Matrix.zero(n + 1, n);
        objective_minimum_finder.add_from(this.tableau.objective,
            {start: k, end: k + n    , map: (index) => index - k},
            {start: k, end: k + n + 1, map: (index) => index - k},
        );
        let minimum_reductor: Sparse.MaybeConsistentReductor =
            new Sparse.GaussianReductor(objective_minimum_finder);
        if (minimum_reductor.consistent) {
            let entering_vector = minimum_reductor.recover_vector(
                Vector.zero(minimum_reductor.image_width) );
            if (entering_vector.every_nonzero(() => false))
                throw new Error("unreachable");
            return {
                entering_vector_base: entering_vector,
                entering_obj_limit: 1,
            };
        }
        let objective_decrease_finder = Matrix.zero(n + 1, n + 1);
        objective_decrease_finder.add_from( this.tableau.objective,
            {start: k, end: k + n + 1, map: (index) => index - k},
            {start: k, end: k + n    , map: (index) => index - k},
        )
        objective_decrease_finder.set_value(n, n, 1);
        let decrease_reductor: Sparse.MaybeConsistentReductor =
            new Sparse.GaussianReductor(objective_decrease_finder);
        if (decrease_reductor.consistent) {
            let entering_vector = decrease_reductor.recover_vector(
                Vector.zero(decrease_reductor.image_width) );
            if (entering_vector.every_nonzero(() => false))
                throw new Error("unreachable");
            return {
                entering_vector_base: entering_vector,
                entering_obj_limit: Infinity,
            };
        }
        throw new Error("unreachable");
    }

    _step_exit = LPProblemSolver.prototype._step_exit;

}

} // end namespace

} // end namespace

