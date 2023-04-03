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
    var model: LPModel = {
        optimize: "choice",
        opType: "max",
        constraints: {},
        variables: {},
    };
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
            var_a = var_base + "_a",
            var_b = var_base + "_b",
            var_c = var_base + "_c",
            var_len = var_base + "_len";
        model.variables[var_a] = {
            choice: Chooser.choose_weight(),
            constr_a_x: direction.x,
            constr_a_y: direction.y,
            [var_len]: 1,
        };
        model.variables[var_b] = {
            choice: Chooser.choose_weight(),
            constr_b_x: direction.x,
            constr_b_y: direction.y,
            [var_len]: 1,
        };
        model.variables[var_c] = {
            choice: Chooser.choose_weight(),
            constr_c_x: direction.x,
            constr_c_y: direction.y,
            [var_len]: 1,
        };
        model.constraints[var_len] = {
            equal: Math.abs(vector.value),
        };
        flow_params[i] = {k, var_a, var_b, var_c};
    }
    let
        {x: constr_a_x, y: constr_a_y} = Graphs.Vector.from_points(
            sector_start, region.points1[0] ),
        {x: constr_b_x, y: constr_b_y} = Graphs.Vector.from_points(
            region.points1[1], region.points2[0] ),
        {x: constr_c_x, y: constr_c_y} = Graphs.Vector.from_points(
            region.points2[1], sector_end );
    Object.assign(model.constraints, {
        constr_a_x: {equal: constr_a_x},
        constr_a_y: {equal: constr_a_y},
        constr_b_x: {equal: constr_b_x},
        constr_b_y: {equal: constr_b_y},
        constr_c_x: {equal: constr_c_x},
        constr_c_y: {equal: constr_c_y},
    });

    var results = lpsolve(model);
    if (!results.feasible) {
        let error: any = new ImpossibleFlowError("Cannot find flow");
        error.info = {find_flows: {uncut_region: region, model, results}};
        throw error;
    }
    var flows: Flows = new Map();
    for (let [i, vector] of vectors.entries()) {
        let {k, var_a, var_b, var_c} = flow_params[i];
        let
            a: number,
            b: number,
            c: number;
        ({[var_a]: a = 0, [var_b]: b = 0, [var_c]: c = 0} = results);
        flows.set( vector.direction,
            {a: a*k, b: b*k, c: c*k, sum: (a+b+c)*k} );
    }
    return flows;
}

