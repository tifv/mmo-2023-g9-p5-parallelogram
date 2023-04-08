class ImpossibleFlowError extends Error {}

class UncutRegion {
    polygon: Polygon;
    triangle1: Polygon;
    triangle2: Polygon;
    constructor(
        polygon: Polygon,
        triangle1: Polygon,
        triangle2: Polygon
    ) {
        this.polygon = polygon;
        this.triangle1 = triangle1;
        this.triangle2 = triangle2;
    }
    get points1(): [Point, Point] {
        return [this.triangle1.vertices[0], this.triangle1.vertices[1]];
    }
    get points2(): [Point, Point] {
        return [this.triangle2.vertices[2], this.triangle2.vertices[0]];
    }
}

type FlowDirections = {
    sector_start: Point,
    sector_end: Point,
    vectors: Array<DirectedVector>
}

function select_flowing_sector(region: UncutRegion): FlowDirections {
    let [, {vector: diagonal}, ] = region.triangle1.oriented_edges();
    let {
        forward:  {vertex: sector_start, index: start_index},
        backward: {vertex: sector_end,   index: end_index},
    } = region.polygon.find_tangent_points(diagonal.opposite());
    if (
        sector_start === sector_end ||
        modulo(end_index - start_index, region.polygon.edges.length) !=
            region.polygon.edges.length / 2
    ) {
        let error:any = new Error(
            "Sector boundaries are not exactly opposite" );
        error.info = {select_flowing_sector: {start_index, end_index}};
        throw error;
    }
    let [{vector: antiflow1}, , ] = region.triangle1.oriented_edges()
    let [, , {vector: antiflow2}] = region.triangle2.oriented_edges()
    let vectors = region.polygon.slice_oriented_edges(start_index, end_index)
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
    return {sector_start, sector_end, vectors};
}

type Flow = {a: number, b: number, c: number, sum: number};
type Flows = Map<Direction, Flow>;

function find_flows(region: UncutRegion, flow_directions: FlowDirections): Flows {
    let {sector_start, sector_end, vectors} = flow_directions;
    type Tokens = Algebra.Expression.Tokens;
    let constraints: {
        a: {x: Tokens, y: Tokens},
        b: {x: Tokens, y: Tokens},
        c: {x: Tokens, y: Tokens},
        other: Array<Tokens>,
    } = {
        a: {x: [], y: []},
        b: {x: [], y: []},
        c: {x: [], y: []},
        other: [],
    };
    let objective: Tokens = [];
    var flow_params: Array<{
        k: number;
        var_a: string; var_b: string; var_c: string;
    }> = [];
    for (let [i, vector] of vectors.entries()) {
        let direction: Graphs.AbstractVector, k: number;
        if (vector.value >= 0) {
            direction = vector.direction;
            k = 1;
        } else {
            direction = vector.direction.opposite();
            k = -1;
        }
        let
            var_base = "k" + i,
            var_a = var_base + ".a",
            var_b = var_base + ".b",
            var_c = var_base + ".c";
        constraints.a.x.push("+", direction.x, "*", var_a);
        constraints.a.y.push("+", direction.y, "*", var_a);
        constraints.b.x.push("+", direction.x, "*", var_b);
        constraints.b.y.push("+", direction.y, "*", var_b);
        constraints.c.x.push("+", direction.x, "*", var_c);
        constraints.c.y.push("+", direction.y, "*", var_c);
        objective.push(
            "+", Chooser.choose_weight(), "*", var_a,
            "+", Chooser.choose_weight(), "*", var_b,
            "+", Chooser.choose_weight(), "*", var_c,
        );
        constraints.other.push(
            [var_a, "+", var_b, "+", var_c, "==", Math.abs(vector.value)],
            [var_a, ">=", 0], [var_b, ">=", 0], [var_c, ">=", 0],
        );
        flow_params[i] = {k, var_a, var_b, var_c};
    }
    {
        let {x, y} = Graphs.Vector.from_points(
            sector_start, region.points1[0] );
        constraints.a.x.push("==", x);
        constraints.a.y.push("==", y);
    }
    {
        let {x, y} = Graphs.Vector.from_points(
            region.points1[1], region.points2[0] );
        constraints.b.x.push("==", x);
        constraints.b.y.push("==", y);
    }
    {
        let {x, y} = Graphs.Vector.from_points(
            region.points2[1], sector_end );
        constraints.c.x.push("==", x);
        constraints.c.y.push("==", y);
    }

    var result = Algebra.Solver.LPProblemSolver.solve_from_tokens({
        objective: objective,
        target: "min",
        constraints: [
            constraints.a.x, constraints.a.y,
            constraints.b.x, constraints.b.y,
            constraints.c.x, constraints.c.y,
            ...constraints.other,
        ],
    })
    if (result.error) {
        let error: any = new ImpossibleFlowError("Cannot find flow");
        error.info = {find_flows: { uncut_region: region,
            objective, constraints, result }};
        throw error;
    }
    var flows: Flows = new Map();
    for (let [i, vector] of vectors.entries()) {
        let {k, var_a, var_b, var_c} = flow_params[i];
        let {[var_a]: a = 0, [var_b]: b = 0, [var_c]: c = 0} = result.solution;
        flows.set( vector.direction,
            {a: a*k, b: b*k, c: c*k, sum: (a+b+c)*k} );
    }
    return flows;
}

