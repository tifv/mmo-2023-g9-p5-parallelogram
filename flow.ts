namespace FlowFinder {

class ImpossibleFlowError extends Error {}

export type Flow = {a: number, b: number, c: number, sum: number};
export type Flows = Map<Direction, Flow>;

import Tokens = Algebra.Expression.Tokens;
import Token  = Algebra.Expression.Token;

type FlowName = "a" | "b" | "c";
let FlowNames: FlowName[] = ["a", "b", "c"];
type EndName = "start" | "end";
let EndNames: EndName[] = ["start", "end"];
type CoordName = "x" | "y";
let CoordNames: CoordName[] = ["x", "y"];

function map_flows<V>(f: (a: FlowName) => V): Record<FlowName,V> {
    return {a: f("a"), b: f("b"), c: f("c")};
}
function map_ends<V>(f: (a: EndName) => V): Record<EndName,V> {
    return {start: f("start"), end: f("end")};
}
function map_coords<V>(f: (a: CoordName) => V): Record<CoordName,V> {
    return {x: f("x"), y: f("y")};
}

export class UncutRegion {
    polygon: Polygon;
    triangle1: Polygon;
    triangle2: Polygon;
    sector_start: Point;
    sector_end: Point;
    flow_vectors: Array<DirectedVector>;

    constructor(
        polygon: Polygon,
        triangle1: Polygon,
        triangle2: Polygon
    ) {
        this.polygon = polygon;
        this.triangle1 = triangle1;
        this.triangle2 = triangle2;
        ({
            sector_start: this.sector_start,
            sector_end  : this.sector_end,
            flow_vectors: this.flow_vectors,
        } = this._find_flow_sector());
    }

    _find_flow_sector() {
        let [, {vector: diagonal}, ] = this.triangle1.oriented_edges();
        let {
            forward:  {vertex: sector_start, index: start_index},
            backward: {vertex: sector_end,   index: end_index},
        } = this.polygon.find_tangent_points(diagonal.opposite());
        if (
            sector_start === sector_end ||
            modulo(end_index - start_index, this.polygon.edges.length) !=
                this.polygon.edges.length / 2
        ) {
            let error:any = new Error(
                "Sector boundaries are not exactly opposite" );
            error.info = {select_flowing_sector: {start_index, end_index}};
            throw error;
        }
        // return {sector_start, start_index, sector_end, end_index};
        let [{vector: antiflow1}, , ] = this.triangle1.oriented_edges()
        let [, , {vector: antiflow2}] = this.triangle2.oriented_edges()
        let flow_vectors = this.polygon
            .slice_oriented_edges(start_index, end_index)
            .map(({vector}) => {
                if (vector.direction == antiflow1.direction) {
                    vector = new DirectedVector( vector.direction,
                        vector.value - antiflow1.value );
                }
                if (vector.direction == antiflow2.direction) {
                    vector = new DirectedVector( vector.direction,
                        vector.value - antiflow2.value );
                }
                return vector;
            });
        return {sector_start, sector_end, flow_vectors};
    }

    get_variable_names(): (
        Record<FlowName,Record<EndName,Record<CoordName,string>>> & {
            flows: (index: number) => Record<FlowName,string>,
        }
    ) {
        return Object.assign( {},
            map_flows(a => map_ends(e => map_coords(x =>
                "flow." + a + "." + e + "." + x
            ))),
            { flows: (index: number) => map_flows( a =>
                "flow:" + index + "." + a
            )},
        );
    }

    _get_flow_points(): Record<FlowName,Record<EndName,Point>> {
        return {
            a: { start: this.sector_start,
                 end:   this.triangle1.vertices[0] },
            b: { start: this.triangle1.vertices[1],
                 end:   this.triangle2.vertices[2] },
            c: { start: this.triangle2.vertices[0],
                 end:   this.sector_end },
        };
    }

    get point1(): Point {
        return this._get_flow_points().b.start;
    }

    set point1(point: Point) {
        let old_point = this.point1;
        this.triangle1 = this.triangle1.shift(
            Vector.between(old_point, point) );
    }

    get point2(): Point {
        return this._get_flow_points().b.end;
    }

    set point2(point: Point) {
        let old_point = this.point2;
        this.triangle2 = this.triangle2.shift(
            Vector.between(old_point, point) );
    }

    get_flow_constraints_base(): Array<Tokens> {
        let var_names = this.get_variable_names();
        let flow_constraints: (
            Record<FlowName,Record<CoordName,Tokens>>
        ) = map_flows(a => map_coords(x => [
                var_names[a].end[x], "-", var_names[a].start[x], "==",
            ]));
        let constraints = new Array<Tokens>();
        for (let [i, vector] of this.flow_vectors.entries()) {
            let
                {direction, value} = vector,
                k = value > 0 ? +1 : -1;
            let names = var_names.flows(i);
            map_flows(a => map_coords(x => {
                flow_constraints[a][x].push("+", direction[x], "*", names[a]);
            }));
            map_flows(a => {
                constraints.push([k, "*", names[a], ">=", 0]);
            });
            constraints.push(
                [value, "==", names.a, "+", names.b, "+", names.c],
            );
        }
        map_flows(a => map_coords(x => {
            constraints.push(flow_constraints[a][x]);
        }));
        return constraints;
    }

    get_flow_constraints_sector(): Array<Tokens> {
        let var_names = this.get_variable_names();
        let constraints = new Array<Tokens>();
        map_coords( x => {
            constraints.push(
                [var_names.a.start[x], "==", this.sector_start[x]],
                [var_names.c.end[x], "==", this.sector_end[x]]
            );
        });
        return constraints;
    }

    get_flow_constraints_triangles_free(): Array<Tokens> {
        let var_names = this.get_variable_names();
        let constraints = new Array<Tokens>();
        let points = this._get_flow_points();
        map_coords( x => {
            constraints.push(
                [ var_names.b.start[x], "-", var_names.a.end[x], "==",
                    points.b.start[x] - points.a.end[x] ],
                [ var_names.c.start[x], "-", var_names.b.end[x], "==",
                    points.c.start[x] - points.b.end[x] ],
            );
        });
        return constraints;
    }

    get_flow_constraints_triangles_fixed(point?: 1 | 2): Array<Tokens> {
        let var_names = this.get_variable_names();
        let constraints = new Array<Tokens>();
        let points = this._get_flow_points();
        map_coords( x => {
            if (point === undefined || point === 1)
                constraints.push(
                    [var_names.b.start[x], "==", points.b.start[x]],
                );
            if (point === undefined || point === 2)
                constraints.push(
                    [var_names.b.end[x], "==", points.b.end[x]],
                );
        });
        return constraints;
    }

    find_flows(): Flows {
        let var_names = this.get_variable_names();
        let constraints = [
            ...this.get_flow_constraints_base(),
            ...this.get_flow_constraints_sector(),
            ...this.get_flow_constraints_triangles_free(),
            ...this.get_flow_constraints_triangles_fixed(),
        ];
        let objective = new Array<Token>();
        for (let [i, vector] of this.flow_vectors.entries()) {
            let value = vector.value;
            let names = var_names.flows(i);
            map_flows(a => {
                objective.push(
                    "+", Chooser.choose_weight(), "*", names[a], "/", value );
            });
        }

        let result = Algebra.Solver.LPProblemSolver.solve_from_tokens({
            objective: objective,
            target: "min",
            constraints,
        })
        if (result.error) {
            let error: any = new ImpossibleFlowError("Cannot find flow");
            error.info = {find_flows: { uncut_region: this,
                objective, constraints, result }};
            throw error;
        }
        let solution = result.solution;
        let flows: Flows = new Map();
        for (let [i, vector] of this.flow_vectors.entries()) {
            let names = var_names.flows(i);
            let values = map_flows( a => {
                let {[names[a]]: x = 0} = solution; return x; });
            flows.set(vector.direction, Object.assign( values,
                {sum: values.a + values.b + values.c} ));
        }
        return flows;
    }

    find_nearest_feasible(
        point1_obj: {exact?: false, close: Point} | {exact: true},
        point2_obj: {exact?: false, close: Point} | {exact: true},
    ): {point1: Point, point2: Point, flows: Flows} {
        let
            var_names = this.get_variable_names(),
            point1_names = var_names.b.start,
            point2_names = var_names.b.end;
        let constraints = [
            ...this.get_flow_constraints_base(),
            ...this.get_flow_constraints_sector(),
            ...this.get_flow_constraints_triangles_free(),
            // ...this.get_flow_constraints_triangles_fixed(),
        ];
        let objective = new Array<Token>();
        if (point1_obj.exact) {
            constraints.push(
                ...this.get_flow_constraints_triangles_fixed(1),
            );
        } else {
            map_coords(x => {
                let diff: Tokens = [
                    "(", point1_names[x], "-", point1_obj.close[x], ")" ];
                objective.push( "+", ...diff, "*", ...diff);
            });
        }
        if (point2_obj.exact) {
            constraints.push(
                ...this.get_flow_constraints_triangles_fixed(2),
            );
        } else {
            map_coords(x => {
                let diff: Tokens = [
                    "(", point2_names[x], "-", point2_obj.close[x], ")" ];
                objective.push( "+", ...diff, "*", ...diff);
            });
        }

        let result = Algebra.Solver.QPProblemSolver.solve_from_tokens({
            objective: objective,
            target: "min",
            constraints,
        })
        if (result.error) {
            let error: any = new ImpossibleFlowError("Cannot find flow");
            error.info = {find_flows: { uncut_region: this,
                objective, constraints, result }};
            throw error;
        }
        let solution = result.solution;
        let flows: Flows = new Map();
        for (let [i, vector] of this.flow_vectors.entries()) {
            let names = var_names.flows(i);
            let values = map_flows( a => {
                let {[names[a]]: x = 0} = solution; return x; });
            flows.set(vector.direction, Object.assign( values,
                {sum: values.a + values.b + values.c} ));
        }
        let point1 = new Point(
            solution[point1_names.x], solution[point1_names.y] );
        let point2 = new Point(
            solution[point2_names.x], solution[point2_names.y] );
        return {point1, point2, flows};
    }

}

} // end namespace

import UncutRegion = FlowFinder.UncutRegion;
import Flows = FlowFinder.Flows;
import Flow = FlowFinder.Flow;

