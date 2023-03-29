const EPSILON = 1e-7;

function epsilon_sign(value: number): number {
    return (value > EPSILON ? 1 : 0) - (value < -EPSILON ? 1 : 0);
}

function get_or_die<K,V>(map: Map<K,V>, key: K): V {
    let value = map.get(key);
    if (value === undefined)
        throw new Error("The key does not belong to the map")
    return value;
}

class NumberSet extends Array<number> {
    static from_numbers(numbers: Iterable<number>): NumberSet {
        let set = new NumberSet();
        for (let number of numbers) {
            set.add(number);
        }
        return set;
    }
    add(number: number): number {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let current = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return current;
            }
        }
        this.splice(min_index, 0, number);
        return number;
    }
    min(): number {
        return this[0];
    }
    max(): number {
        return this[this.length - 1];
    }
}

class NumberMap<V> extends Array<[number,V]> {
    set(number: number, value: V): [number, V] {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return [current, current_value];
            }
        }
        this.splice(min_index, 0, [number, value]);
        return [number, value];
    }
    get(number: number): V {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return current_value;
            }
        }
        throw new Error("Number is not in the map.");
    }
    index_of_key(number: number): number {
        let
            min_index = 0, max_index = this.length,
            index = min_index;
        while(max_index > min_index) {
            index = Math.floor((min_index + max_index) / 2);
            let [current, current_value] = this[index];
            if (current > number + EPSILON) {
                max_index = index;
            } else if (current < number - EPSILON) {
                min_index = index + 1;
            } else {
                return index;
            }
        }
        throw new Error("Number is not in the map.");
    }
    index_of_value(value: V): number {
        for (let i = 0; i < this.length; ++i) {
            let [, current_value] = this[i];
            if (current_value === value)
                return i;
        }
        throw new Error("Value is not in the map.");
    }
}

class SlicedArray<V> extends NumberMap<V[]> {
    add_guard(number: number) {
        this.push([number, []]);
    }
    slice_values(start?: number, end?: number): Array<V> {
        let start_index: number = start !== undefined ? 
            this.index_of_key(start) : 0;
        let sliced_values = new Array<V>();
        for (let i = start_index; i < this.length; ++i) {
            let [current, values] = this[i];
            if (end !== undefined) {
                if (current == end)
                    break;
                if (current > end)
                    throw new Error("End number is not in the map.");
            }
            sliced_values.push(...values);
        }
        return sliced_values;
    }
}

class ImpossibleConstructError extends Error {}

class Pair {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
    *[Symbol.iterator] () {
        yield this.x;
        yield this.y;
    }
}

class AbstractVector extends Pair {
    opposite(): AbstractVector {
        return new AbstractVector(-this.x, -this.y);
    }
    project(direction: Direction) {
        return this.x * direction.x + this.y * direction.y;
    }
    skew(vector: AbstractVector) {
        return this.y * vector.x - this.x * vector.y;
    }
}

class Direction extends AbstractVector {
    constructor(x: number, y: number) {
        let length = (x**2 + y**2)**(1/2);
        super(x / length, y / length);
    }
    static from_angle(angle: number) {
        return new Direction(Math.cos(angle), Math.sin(angle));
    }
}

class Vector extends AbstractVector {
    static from_points(start: Point, end: Point) {
        return new Vector(end.x - start.x, end.y - start.y);
    }
    get length() {
        return (this.x**2 + this.y**2)**(1/2);
    }
    opposite(): Vector {
        return new Vector(-this.x, -this.y);
    }
}

class DirectedVector extends Vector {
    direction: Direction;
    value: number;
    constructor(direction: Direction, value: number) {
        super(direction.x * value, direction.y * value)
        this.direction = direction;
        this.value = value;
    }
    opposite(): DirectedVector {
        return new DirectedVector(this.direction, -this.value);
    }
    static from_collinear(
        direction: Direction, vector: Vector,
    ): DirectedVector {
        return new DirectedVector(direction, vector.project(direction));
    }
}

class Point extends Pair {
    shift(vector: Vector): Point {
        return new Point(this.x + vector.x, this.y + vector.y);
    }
    is_equal(point: Point) {
        return ( Math.abs(this.x - point.x) < EPSILON &&
            Math.abs(this.y - point.y) < EPSILON );
    }
}


class Edge {
    start: Point;
    delta: DirectedVector;
    end: Point;

    constructor(start: Point, delta: DirectedVector, end: Point) {
        if (delta.value < 0)
            [start, delta, end] = [end, delta.opposite(), start];
        this.start = start;
        this.delta = delta;
        this.end = end;
    }

    *[Symbol.iterator] (): Generator<Point, void, undefined> {
        yield this.start;
        yield this.end;
    }
    incident_to(vertex: Point) {
        return vertex === this.start || vertex === this.end;
    }

    get direction(): Direction {
        return this.delta.direction;
    }

    static from(start: Point, delta: DirectedVector): Edge {
        return new Edge(start, delta, start.shift(delta));
    }

    other_end(vertex: Point): Point {
        if (vertex === this.start)
            return this.end;
        if (vertex === this.end)
            return this.start;
        throw new Error("The point is not incident to this edge");
    }
    replace_vertex_obj(old_vertex: Point, new_vertex: Point): Edge {
        if (old_vertex === this.start)
            return new Edge(new_vertex, this.delta, this.end);
        if (old_vertex === this.end)
            return new Edge(this.start, this.delta, new_vertex);
        throw new Error("The point is not incident to this edge");
    }
    delta_from(vertex: Point): DirectedVector {
        if (vertex === this.start)
            return this.delta;
        if (vertex === this.end)
            return this.delta.opposite();
        throw new Error("The point is not incident to this edge");
    }

    shift(vector: Vector): Edge {
        return new Edge(
            this.start.shift(vector), this.delta, this.end.shift(vector) );
    }

    split_from(start: Point, values: number[]): Array<Edge> {
        let vertex = start;
        let edges: Edge[] = [];
        let rest_value = this.delta_from(vertex).value;
        for (let value of values) {
            let
                delta = new DirectedVector(this.direction, value),
                end = vertex.shift(delta),
                edge = new Edge(vertex, delta, end);
            edges.push(edge);
            vertex = end;
            rest_value -= value;
        }
        {
            let
                delta = new DirectedVector(this.direction, rest_value),
                last_edge = new Edge(vertex, delta, this.other_end(start));
            edges.push(last_edge);
        }
        return edges;
    }

    substitute(
        vertex_map: (vertex: Point) => Point,
    ): Edge {
        let
            start = vertex_map(this.start),
            end = vertex_map(this.end);
        if (start === this.start && end === this.end)
            return this;
        return new Edge(start, this.delta, end);
    }
}

type OrientedEdge = {
    edge: Edge, index: number,
    forward: boolean, vector: DirectedVector,
}

class Polygon {
    edges: Array<Edge>;
    size: number;
    vertices: Array<Point>;
    constructor(start: Point, edges: Array<Edge>) {
        this.edges = edges;
        this.size = edges.length;
        this.vertices = [];
        if (this.edges.length == 0) {
            this.vertices.push(start);
            return;
        }

        let vertex = start;
        for (let edge of this.edges) {
            this.vertices.push(vertex);
            vertex = edge.other_end(vertex);
        }
        if (vertex !== start)
            throw new Error("Polygon is not cyclic");
    }
    /** This is the only modifying operation on Polygon.
     * It only affects iteration order of its edges. */
    rotate(new_start: Point) {

    }

    *[Symbol.iterator] (): Generator<Edge, void, undefined> {
        yield* this.edges;
    }
    *oriented_edges(): Generator<OrientedEdge, void, undefined> {
        let vertex = this.vertices[0];
        for (let [index, edge] of this.edges.entries()) {
            if (edge.start === vertex) {
                yield { edge, index,
                    forward: true, vector: edge.delta };
            } else {
                yield { edge, index,
                    forward: false, vector: edge.delta.opposite() };
            }
            vertex = edge.other_end(vertex);
        }
    }
    _index_modulo(index: number): number {
        var n = this.edges.length;
        return ((index % n) + n) % n;
    }
    get_edge(index: number): Edge {
        index = this._index_modulo(index);
        return this.edges[index];
    }
    slice_edges(
        start_index: number,
        end_index: number,
        {allow_empty = false} = {},
    ): Array<Edge> {
        start_index = this._index_modulo(start_index);
        end_index = this._index_modulo(end_index);
        if ( allow_empty && end_index >= start_index ||
            end_index > start_index )
        {
            return this.edges.slice(start_index, end_index);
        }
        return this.edges.slice(start_index).concat(
            this.edges.slice(0, end_index) );
    }
    get_vertex(index: number): Point {
        if (this.edges.length == 0)
            return this.vertices[0];
        index = this._index_modulo(index);
        return this.vertices[index];
    }

    reversed(): Polygon {
        return new Polygon(this.vertices[0], Array.from(this.edges).reverse());
    }

    substitute(
        vertex_map: (vertex: Point) => Point,
        edge_map: (edge: Edge) => (Edge | Edge[]),
    ): Polygon {
        let changes = false;
        function* new_edges(
            oriented_edges: Generator<OrientedEdge, void, undefined>,
        ): Generator<Edge, void, undefined> {
            for (let {edge, forward} of oriented_edges) {
                let replacement = edge_map(edge);
                if (replacement === edge) {
                    yield edge;
                    continue;
                }
                changes = true;
                if (replacement instanceof Edge) {
                    yield replacement;
                    continue;
                }
                if (forward) {
                    replacement = Array.from(replacement).reverse();
                }
                yield* replacement;
            }
        }
        let start = vertex_map(this.vertices[0]);
        if (start !== this.vertices[0])
            changes = true;
        if (!changes)
            return this;
        return new Polygon( start,
            Array.from(new_edges(this.oriented_edges())) );
    }


    find_tangent_point(vector: AbstractVector): {
        forward: { vertex: Point; index: number; };
        backward: { vertex: Point; index: number; };
    } {
        if (this.edges.length == 0) {
            let [vertex] = this.vertices;
            return {
                forward: {vertex: vertex, index: 0},
                backward: {vertex: vertex, index: 0},
            };
        }
        var lefts = this.edges.map( edge =>
            epsilon_sign(edge.delta.skew(vector)) );
        let
            forward_indices: Array<number> = [],
            backward_indices: Array<number> = [];
        for (let i = 0; i < this.edges.length; ++i) {
            if ( lefts[this._index_modulo(i-1)] <= 0 &&
                lefts[this._index_modulo(i)] > 0 )
            {
                forward_indices.push(i);
            }
            if ( lefts[this._index_modulo(i-1)] >= 0 &&
                lefts[this._index_modulo(i)] < 0 )
            {
                backward_indices.push(i);
            }
        }
        if (forward_indices.length < 1 || backward_indices.length < 1 ) {
            throw new Error("Couldn't find a tangent");
        }
        if (forward_indices.length > 1 || backward_indices.length > 1) {
            throw new Error("Found multiple tangents somehow");
        }
        let [i] = forward_indices, [j] = backward_indices;
        return {
            forward:  {vertex: this.get_vertex(i), index: i},
            backward: {vertex: this.get_vertex(j), index: j},
        };
    }

    static from_vectors(origin: Point, vectors: Array<DirectedVector>):
        Polygon
    {
        let last_point = origin;
        var edges: Array<Edge> = [];
        for (let vector of vectors) {
            let edge = Edge.from(last_point, vector);
            last_point = edge.end;
            edges.push(edge);
        }
        let n = edges.length;
        let last_edge = edges[n-1] =
            edges[n-1].replace_vertex_obj(last_point, origin);
        if (!last_edge.end.is_equal(
                last_edge.start.shift(last_edge.delta)
        )) {
            console.log(edges);
            throw new Error("Polygon has not cycled correctly");
        }
        return new Polygon(origin, edges);
    }

    static make_regular(n: number, side_length: number = 1.0):
        {polygon: Polygon, directions: Direction[]}
    {
        let directions: Array<Direction> = [];
        let vectors: Array<DirectedVector> = [];
        for (let i = 0; i < n; ++i) {
            let direction = Direction.from_angle((Math.PI/n) * i);
            directions.push(direction);
            vectors.push(new DirectedVector(direction, side_length));
        }
        for (let i = 0; i < n; ++i) {
            let direction = directions[i];
            vectors.push(new DirectedVector(direction, -side_length));
        }
        return {
            polygon: Polygon.from_vectors(new Point(0, 0), vectors),
            directions: directions,
        };
    }

}

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
    get point1(): Point {
        return this.triangle1.vertices[1];
    }
    get point2(): Point {
        return this.triangle2.vertices[2];
    }
}

type EdgeSet = Array<Edge>;

/** First element is the left face (face with forward direction of the edge),
 *  second is the right face. */
type FaceSet = [Polygon | null, Polygon | null];

class PlanarGraph {
    vertices: Set<Point>;
    edges: Set<Edge>;
    edgemap: Map<Point,EdgeSet>;
    faces: Set<Polygon>;
    facemap: Map<Edge,FaceSet>;
    constructor(
        vertices: Iterable<Point>,
        edges: Iterable<Edge>,
        faces: Iterable<Polygon>,
    ) {
        this.vertices = new Set(vertices);
        this.edges = new Set(edges);
        this.edgemap = PlanarGraph._build_edgemap(this.edges);
        this.faces = new Set(faces);
        this.facemap = PlanarGraph._build_facemap(this.faces);
    }
    static _build_edgemap(edges: Iterable<Edge>): Map<Point,EdgeSet> {
        let edgemap: Map<Point,EdgeSet> = new Map();
        for (let edge of edges) {
            for (let vertex of edge) {
                let edgeset = edgemap.get(vertex);
                if (edgeset === undefined) {
                    edgeset = [];
                    edgemap.set(vertex, edgeset);
                }
                edgeset.push(edge);
            }
        }
        return edgemap;
    }
    vertex_edges(vertex: Point): EdgeSet {
        let edgeset = this.edgemap.get(vertex);
        if (edgeset === undefined)
            throw new Error("Graph data structure is compromised");
        return edgeset;
    }
    static _build_facemap(faces: Iterable<Polygon>): Map<Edge,FaceSet> {
        let facemap: Map<Edge,FaceSet> = new Map();
        for (let face of faces) {
            for (let {edge, forward} of face.oriented_edges()) {
                let faceset = facemap.get(edge);
                if (faceset === undefined) {
                    faceset = [null, null];
                    facemap.set(edge, faceset);
                }
                if (forward) {
                    if (faceset[0] !== null)
                        throw new Error(
                            "Face with forward direction is already present" );
                    faceset[0] = face;
                } else {
                    if (faceset[1] !== null)
                        throw new Error(
                            "Face with inverse direction is already present" );
                    faceset[1] = face;
                }
            }
        }
        return facemap;
    }
    edge_faces(edge: Edge): FaceSet {
        let faceset = this.facemap.get(edge);
        if (faceset === undefined)
            throw new Error("PlanarGraph data structure is compromised");
        return faceset;
    }
    copy(): PlanarGraph {
        return new PlanarGraph(this.vertices, this.edges, this.faces);
    }
    substitute(
        vertex_map: (vertex: Point) => Point,
        edge_map: (edge: Edge) => (Edge | Edge[]),
    ): PlanarGraph {

        let vertices: Point[] = [];
        for (let vertex of this.vertices) {
            vertices.push(vertex_map(vertex));
        }

        let edges: Edge[] = [];
        for (let edge of this.edges) {
            let replacement = edge_map(edge);
            if (replacement instanceof Edge) {
                edges.push(replacement);
                continue;
            }
            let start = vertex_map(edge.start), vertex = start;
            let end = vertex_map(edge.end);
            let new_vertices: Point[] = [];
            for (let replacement_edge of replacement) {
                edges.push(replacement_edge);
                vertex = replacement_edge.other_end(vertex);
                new_vertices.push(vertex);
            }
            if (new_vertices.pop() !== end) {
                throw new Error("Edge replacement is not correct");
            }
            vertices.push(...new_vertices);
        }

        let faces: Polygon[] = [];
        for (let face of this.faces) {
            faces.push(face.substitute(vertex_map, edge_map));
        }
        return new PlanarGraph(vertices, edges, faces);
    }
}

class CutRegion {
    graph: PlanarGraph;
    outer_face: Polygon;
    triangle1: Polygon;
    triangle2: Polygon;
    constructor(
        graph: PlanarGraph,
        outer_face: Polygon,
        triangle1: Polygon,
        triangle2: Polygon,
    ) {
        this.graph = graph;
        this.outer_face = outer_face;
        this.triangle1 = triangle1;
        this.triangle2 = triangle2;
    }
    get_polygon() {
        return this.outer_face.reversed();
    }

    static initial(
        origin: Point,
        [vec1, vec2, vec3]:
            [DirectedVector, DirectedVector, DirectedVector] ,
    ): CutRegion {
        let polygon = Polygon.from_vectors(origin, [
            vec1, vec3.opposite(), vec1.opposite(), vec3,
        ]);
        let outer_face = polygon.reversed();
        let diagonal = new Edge(
            polygon.vertices[1], vec2, polygon.vertices[3] );
        let triangle1 = new Polygon( origin,
            [polygon.edges[0], diagonal, polygon.edges[3]] );
        let triangle2 = new Polygon( polygon.vertices[2],
            [polygon.edges[2], diagonal, polygon.edges[1]] );
        return new CutRegion(
            new PlanarGraph(
                polygon.vertices,
                [...polygon.edges, diagonal],
                [outer_face, triangle1, triangle2],
            ),
            outer_face,
            triangle1, triangle2,
        );
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
    } = region.polygon.find_tangent_point(diagonal);
    if (end_index != start_index + region.polygon.edges.length / 2) {
        throw new Error("Some problems with indices or something");
    }
    let vectors = region.polygon.slice_edges(start_index, end_index)
        .map(edge => edge.delta);
    return {sector_start, sector_end, vectors};
}

type Flow = {a: number, b: number, c: number, sum: number};
type Flows = Map<Direction, Flow>;

type LPModel = {
    optimize: any,
    opType: any,
    variables: {[x :string]: {[x: string]: number}},
    constraints: { [x: string]:
        {max?: number, min?: number, equal?: number}
    },
};

function lpsolve(model: LPModel): { [s: string]: any; } {
    // @ts-ignore
    return solver.solve(model);
}

function find_flows(region: UncutRegion, flow_directions: FlowDirections): Flows {
    let {sector_start, sector_end, vectors} = flow_directions;
    var model: LPModel = {
        optimize: "nothing",
        opType: "max",
        constraints: {},
        variables: {},
    };
    var flow_params: Array<{
        k: number;
        var_a: string; var_b: string; var_c: string;
    }> = [];
    for (let [i, vector] of vectors.entries()) {
        let direction: AbstractVector, k: number;
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
            constr_a_x: direction.x,
            constr_a_y: direction.y,
            [var_len]: 1,
        };
        model.variables[var_b] = {
            constr_b_x: direction.x,
            constr_b_y: direction.y,
            [var_len]: 1,
        };
        model.variables[var_c] = {
            constr_c_x: direction.x,
            constr_c_y: direction.y,
            [var_len]: 1,
        };
        model.constraints[var_len] = {
            equal: 1,
        };
        flow_params[i] = {k, var_a, var_b, var_c};
    }
    let
        {x: constr_a_x, y: constr_a_y} = Vector.from_points(
            flow_directions.sector_start, region.point1 ),
        {x: constr_b_x, y: constr_b_y} = Vector.from_points(
            region.point1, region.point2 ),
        {x: constr_c_x, y: constr_c_y} = Vector.from_points(
            region.point2, flow_directions.sector_end );
    Object.assign(model.constraints, {
        constr_a_x, constr_a_y,
        constr_b_x, constr_b_y,
        constr_c_x, constr_c_y,
    });

    var results = lpsolve(model);
    if (!results.feasible) {
        throw new ImpossibleConstructError("Cannot find flow");
    }
    var flows: Flows = new Map();
    for (let [i, vector] of vectors.entries()) {
        let {k, var_a, var_b, var_c} = flow_params[i];
        let
            a: number,
            b: number,
            c: number;
        ({[var_a]: a, [var_b]: b, [var_c]: c} = results);
        if (
            !(typeof a !== "number") || !(typeof b !== "number") ||
            !(typeof c !== "number")
        ) {
            throw new Error("Solver did not return value");
        }
        flows.set( vector.direction,
            {a: a*k, b: b*k, c: c*k, sum: (a+b+c)*k} );
    }
    return flows;
}

function construct_cut_region(uncut_region: UncutRegion, flows: Flows) {
    let origin = new Point(0, 0);
    let [vec1, vec2, vec3] = Array.from(
        uncut_region.triangle1.oriented_edges()
    ).map(({vector}) => vector);
    let region = CutRegion.initial(origin, [vec1, vec2, vec3]);
    var sector_start = origin, sector_end = origin;
    for (let [direction, flow] of flows) {
        ({region, sector_start, sector_end} =
            Incutter.incut(
                region, sector_start, sector_end,
                direction, flow,
            ));
    }
    return region;
}

type HeightRange = {min: number, max: number, height?: number};

type VertexGroup =
    {vertices: Iterable<Point>, edges: Iterable<Edge>};

class Incutter {
    start_region: CutRegion;
    direction: Direction;
    graph: PlanarGraph;
    polygon: Polygon;
    start: Point;
    end: Point;
    upper_border_edges: Array<Edge>;
    lower_border_edges: Array<Edge>;

    max_height: number;
    heighted_graph: HeightedFaceGraph;
    edge_height_map: Map<Edge, NumberSet>;
    vertex_height_map: Map<Point, NumberSet>;

    vertex_groups: Set<Point|VertexGroup>;

    vertex_image_map: Map<Point,NumberMap<Point>>;
    new_edge_map: Map<Point,SlicedArray<Edge>>;
    edge_image_map: Map<Edge,NumberMap<Edge|Edge[]>>;
    new_face_map: Map<Edge,Polygon[]>;
    face_image_map: Map<Polygon,Polygon>;

    static incut(
        region: CutRegion,
        sector_start: Point,
        sector_end: Point,
        direction: Direction,
        flow: Flow,
    ): {region: CutRegion, sector_start: Point, sector_end: Point} {
        let max_height = Math.abs(flow.sum);
        if (max_height < EPSILON)
            return {region, sector_start, sector_end};
        let incutter = new Incutter( region,
            direction, max_height );
        let {start_height, end_height, height1, height2, heights} =
            incutter.determine_height_values(flow, max_height);
        let heighted_graph = incutter.build_heighted_graph(max_height);
        heighted_graph.set_face_height(region.triangle1, height1);
        heighted_graph.set_face_height(region.triangle2, height2);
        heighted_graph.resolve_iffy_heights();
        incutter.determine_face_heights(heights);
        incutter.determine_edge_heights();
        incutter.determine_vertex_heights();
        incutter.group_vertices();

        incutter.init_image_maps();
        incutter.generate_vertex_images();
        incutter.generate_edge_images();
        incutter.generate_face_images();
        let new_polygon = incutter.regenerate_polygon();
        let new_outer_face = new_polygon.reversed();
        function* all_new_vertices(
            vertex_image_map: Map<Point,NumberMap<Point>>
        ): Generator<Point,void,undefined> {
            for (let [, images] of vertex_image_map) {
                for (let [, image] of images) {
                    yield image;
                }
            }
        }
        function* all_new_edges(
            edge_image_map: Map<Edge,NumberMap<Edge|Edge[]>>,
            new_edge_map: Map<Point,SlicedArray<Edge>>,
        ): Generator<Edge,void,undefined> {
            for (let [, images] of edge_image_map) {
                for (let [, image] of images) {
                    if (image instanceof Edge) {
                        yield image;
                    } else {
                        yield* image;
                    }
                }
            }
            for (let [, edges] of new_edge_map) {
                yield* edges.slice_values();
            }
        }
        function* all_new_faces(
            face_image_map: Map<Polygon,Polygon>,
            new_face_map: Map<Edge,Polygon[]>,
        ): Generator<Polygon,void,undefined> {
            for (let [, image] of face_image_map) {
                yield image;
            }
            for (let [, faces] of new_face_map) {
                yield* faces;
            }
        }
        return {
            region: new CutRegion(
                new PlanarGraph(
                    Array.from(all_new_vertices(incutter.vertex_image_map)),
                    Array.from(all_new_edges(
                        incutter.edge_image_map, incutter.new_edge_map )),
                    Array.from(all_new_faces(
                        incutter.face_image_map, incutter.new_face_map )),
                ),
                new_outer_face,
                get_or_die(incutter.face_image_map, region.triangle1),
                get_or_die(incutter.face_image_map, region.triangle2),
            ),
            sector_start:
                get_or_die(incutter.vertex_image_map, sector_start)
                    .get(start_height),
            sector_end:
                get_or_die(incutter.vertex_image_map, sector_end)
                    .get(end_height),
        };
    }

    constructor(
        region: CutRegion,
        direction: Direction,
        max_height: number,
    ) {
        this.start_region = region;
        this.direction = direction;
        this.max_height = max_height;
        this.graph = region.graph;
        this.polygon = region.get_polygon();
        let start_index: number, end_index: number;
        ({
            forward: { vertex: this.start, index: start_index },
            backward: { vertex: this.end, index: end_index },
        } = this.polygon.find_tangent_point(direction));
        this.upper_border_edges = 
            this.polygon.slice_edges(start_index, end_index)
                .filter(edge => edge.direction !== this.direction );
        this.lower_border_edges = 
            this.polygon.slice_edges(end_index, start_index)
                .filter(edge => edge.direction !== this.direction );
    }
    determine_height_values(flow: Flow, max_height: number):
    {
        start_height: number, end_height: number,
        height1: number; height2: number;
        heights: number[],
    } {
        let [start_height, end_height] = ( flow.sum > 0 ?
            [0, max_height] : [max_height, 0] );
        let height1: number, height2: number;
        let
            a_is_zero = Math.abs(flow.a) < EPSILON,
            b_is_zero = Math.abs(flow.b) < EPSILON,
            c_is_zero = Math.abs(flow.c) < EPSILON;
    
        if (a_is_zero) {
            height1 = 0;
        } else if (b_is_zero && c_is_zero) {
            height1 = max_height;
        } else {
            height1 = start_height + flow.a;
        }
        if (b_is_zero) {
            height2 = height1;
        } else if (c_is_zero) {
            height2 = end_height;
        } else {
            height2 = end_height - flow.c;
        }
        let heights = NumberSet.from_numbers(
            [start_height, height1, height2, end_height] );
        // XXX TODO add more intermediate heights
        return {start_height, end_height, height1, height2, heights};
    }
    build_heighted_graph(max_height: number): HeightedFaceGraph {
        this.heighted_graph = new HeightedFaceGraph(
            this.graph.vertices,
            this.graph.edges,
            function*(faces, outer_face) {
                for (let face of faces) {
                    if (face === outer_face)
                        continue;
                    yield face;
                }
            }(this.graph.faces, this.start_region.outer_face),
            this.direction, max_height,
        );
        return this.heighted_graph;
    }
    determine_face_heights(heights: number[]) {
        while (true) {
            let floating_faces = new Array<Polygon>();
            for (let [face, height] of this.heighted_graph.face_heights) {
                if (height.height === undefined)
                    floating_faces.push(face);
            }
            if (floating_faces.length == 0)
                break;
            let
                floating_face = PRNG.choose_face(floating_faces),
                height = this.heighted_graph.get_face_height(floating_face);
            // XXX set any possible intermediate heights, not just min and max
            this.heighted_graph.set_face_height( floating_face,
                PRNG.choose_element([height.min, height.max]) );
        }
        /**
         * XXX TODO add any underused intermediate heights to vertices
         * and connect them to other objects that use them.
         * (Track usage of each height in operations above.
         * Or the number of occurences.)
         */
    }
    determine_edge_heights() {
        this.edge_height_map = new Map();
        for (let edge of this.heighted_graph.edges) {
            let edge_heights: NumberSet = new NumberSet();
            for (let face of this.heighted_graph.edge_faces(edge)) {
                if (face === null)
                    continue;
                let face_height =
                    this.heighted_graph.get_face_height(face).height;
                if (face_height === undefined)
                    throw new Error("Not all face heights were defined");
                edge_heights.add(face_height);
            }
            this.edge_height_map.set(edge, edge_heights);
        }
        for (let edge of this.lower_border_edges) {
            get_or_die(this.edge_height_map, edge).add(0);
        }
        for (let edge of this.upper_border_edges) {
            get_or_die(this.edge_height_map, edge).add(this.max_height);
        }
    }
    determine_vertex_heights() {
        this.vertex_height_map = new Map();
        for (let vertex of this.heighted_graph.vertices) {
            let heights: NumberSet = new NumberSet();
            for (let edge of this.heighted_graph.vertex_edges(vertex)) {
                let edge_heights = get_or_die(this.edge_height_map, edge);
                for (let height of edge_heights)
                    heights.add(height);
            }
            this.vertex_height_map.set(vertex, heights);
        }
    }
    group_vertices() {
        this.vertex_groups = new Set();
        let group_builder = new Map<Point,VertexGroup>();
        for (let edge of this.heighted_graph.edges) {
            if (edge.direction !== this.direction)
                continue;
            type GroupPart = VertexGroup | undefined;
            let
                group1: GroupPart = group_builder.get(edge.start),
                group2: GroupPart = group_builder.get(edge.end);
            if (group1 === undefined)
                group1 = {vertices: [edge.start], edges: []};
            if (group2 === undefined)
                group2 = {vertices: [edge.end], edges: []};
            let group = {
                vertices: [...group1.vertices, ...group2.vertices],
                edges: [edge, ...group1.edges, ...group2.edges],
            };
            for (let vertex of group.vertices) {
                group_builder.set(vertex, group);
            }
        }
        for (let vertex of this.heighted_graph.vertices) {
            let vertex_group = group_builder.get(vertex);
            if (vertex_group !== undefined) {
                this.vertex_groups.add(vertex_group);
            } else {
                this.vertex_groups.add(vertex);
            }
        }
    }
    init_image_maps() {
        this.vertex_image_map = new Map();
        this.new_edge_map = new Map();
        this.edge_image_map = new Map();
        this.new_face_map = new Map();
        this.face_image_map = new Map();
    }
    generate_vertex_images(): void {
        for (let vertex_group of this.vertex_groups) {
            this.generate_vertex_group_images(
                vertex_group instanceof Point ?
                    {vertices: [vertex_group], edges: []} : vertex_group
            );
        }
    }
    generate_vertex_group_images ({vertices, edges}: VertexGroup): void
    {

        let projection_images = new NumberMap<Point>();
        let vertex_data = new Array<[Point, NumberMap<number>]>();
        for (let vertex of vertices) {
            let heights = get_or_die(this.vertex_height_map, vertex)
            let v_images = new NumberMap<Point>();
            let v_projections = new NumberMap<number>();
            for (let height of heights) {
                let
                    image = height <= EPSILON ? vertex : vertex.shift(
                        new DirectedVector(this.direction, height) ),
                    projection = Vector.from_points(this.start, image)
                        .project(this.direction);
                [projection, image] = projection_images.set(
                    projection, image );
                v_images.set(height, image);
                v_projections.set(height, projection);
            }
            this.vertex_image_map.set(vertex, v_images);
            vertex_data.push([vertex, v_projections]);
        }

        let new_edges = new SlicedArray<Edge>();
        for (let i = 1; i < projection_images.length; ++i) {
            let
                [start_projection, start] = projection_images[i-1],
                [,end] = projection_images[i];
            new_edges.push([start_projection, [new Edge(
                start,
                DirectedVector.from_collinear( this.direction,
                    Vector.from_points(start, end) ),
                end,
            )]]);
        }
        new_edges.add_guard(
            projection_images[projection_images.length-1][0] );

        for (let [vertex, v_projections] of vertex_data) {
            let v_edge_map = new SlicedArray<Edge>();
            for (let i = 1; i < v_projections.length; ++i) {
                let [height, projection] = v_projections[i-1];
                let [, next_projection] = v_projections[i];
                v_edge_map.push([ height,
                    new_edges.slice_values(projection, next_projection) ]);
            }
            this.new_edge_map.set(vertex, v_edge_map);
        }

        for (let edge of edges) {
            let heights = get_or_die(this.edge_height_map, edge);
            let edge_images = new NumberMap<Edge|Edge[]>();
            let [start_images, end_images] = [edge.start, edge.end]
                .map((vertex) => get_or_die(this.vertex_image_map, vertex));
            for (let height of heights) {
                let [start_index, end_index] = [start_images, end_images]
                    .map((image_map) => image_map.get(height))
                    .map((image) => projection_images.index_of_value(image));
                edge_images.set(height,
                    new_edges.slice_values(start_index, end_index) );
            }
            this.edge_image_map.set(edge, edge_images);
        }

    }
    generate_edge_images() {
        for (let [edge, heights] of this.edge_height_map) {
            if (edge.direction === this.direction) {
                if (!this.edge_image_map.has(edge))
                    throw new Error("a vertical edge has been skipped");
                continue;
            }

            let edge_images = new NumberMap<Edge>();
            for (let height of heights) {
                edge_images.set(height, edge.substitute( (vertex: Point) =>
                    get_or_die(this.vertex_image_map, vertex).get(height) ));
            }
            this.edge_image_map.set(edge, edge_images);

            let new_faces = new Array<Polygon>();
            for (let i = 1; i < edge_images.length; ++i) {
                let [height_lower, image_lower] = edge_images[i-1];
                let [height_upper, image_upper] = edge_images[i];
                let [image_start, image_end] = [edge.start, edge.end].map(
                    (vertex) => get_or_die(this.new_edge_map, vertex)
                        .slice_values(height_lower, height_upper)
                );
                let
                    image_left: Edge[], image_right: Edge[],
                    face_start: Point;
                if (edge.direction.skew(this.direction) > 0) {
                    [image_left, image_right] = [image_end, image_start];
                    face_start = image_lower.start;
                } else {
                    [image_left, image_right] = [image_start, image_end];
                    face_start = image_lower.end;
                }
                new_faces.push(new Polygon(face_start, [
                    ...image_right, image_upper,
                    ...image_left.reverse(), image_lower,
                ]));
            }
            this.new_face_map.set(edge, new_faces);
        }
    }
    generate_face_images() {
        for (let [face, {height: maybe_height}] of this.heighted_graph.face_heights) {
            if (maybe_height === undefined) {
                throw new Error("Not all polygons have heights.")
            }
            let height = maybe_height;
            this.face_image_map.set(face, face.substitute(
                (vertex) =>
                    get_or_die(this.vertex_image_map, vertex).get(height),
                (edge) =>
                    get_or_die(this.edge_image_map, edge).get(height),
            ));
        }
    }
    regenerate_polygon(): Polygon {
        type DirectionType = (0|1|2|3);
        type Height = number | "min" | "max";
        let numeric_height: (height: Height) => number
            = (height: Height) => {
                if (height == "min") return 0;
                if (height == "max") return this.max_height;
                return height;
            };
        let get_direction_type: (vector: DirectedVector) => DirectionType
            = (vector: DirectedVector) => {
                return (vector.direction === this.direction)
                    ? (vector.project(this.direction) > 0 ? 0 : 2)
                    : (vector.skew(this.direction) > 0 ? 1 : 3);
            };
        let start_preimage: [Point, Height] | undefined;
        function* edge_generator(
            polygon: Polygon,
            vertex_edge_map: (vertex: Point) =>
                {edges: Array<Edge>, min: number, max: number},
            edge_map: (vertex: Edge, height: Height) =>
                Edge | Array<Edge>,
        ): Generator<Edge, void, undefined> {
            let prev_vector: DirectedVector | null = null;
            for ( let {edge, index, forward, vector}
                of polygon.oriented_edges() )
            {
                let
                    start = forward ? edge.start : edge.end,
                    end = edge.other_end(start);
                if (prev_vector === null) {
                    let prev_edge = polygon.get_edge(index-1);
                    prev_vector = prev_edge.delta_from(
                        prev_edge.other_end(end) );
                }
                let
                    direction_type = get_direction_type(vector),
                    prev_type = get_direction_type(prev_vector);
                let edge_height: Height, start_height: Height;
                if ( direction_type == 0 || direction_type == 2 ||
                    direction_type != prev_type
                ) {
                    let {edges: start_image, min, max} = vertex_edge_map(start);
                    if (direction_type == 0 || direction_type == 1) {
                        edge_height = max;
                        start_height = min;
                    } else {
                        start_image = start_image.reverse();
                        edge_height = min;
                        start_height = max;
                    }
                    yield* start_image;
                } else {
                    if (direction_type == 1) {
                        edge_height = start_height = "max";
                    } else {
                        edge_height = start_height = "min";
                    }
                }
                if (start_preimage === undefined)
                    start_preimage = [start, start_height];
                let edge_image = edge_map(edge, edge_height);
                if (edge_image instanceof Edge) {
                    yield edge_image;
                } else {
                    if (!forward)
                        edge_image = Array.from(edge_image).reverse()
                    yield* edge_image;
                }
                prev_vector = vector;
            }
        }
        let edges = Array.from(edge_generator(
            this.polygon,
            (vertex) => {
                let heights = get_or_die(this.vertex_height_map, vertex);
                return {
                    edges: get_or_die(this.new_edge_map, vertex)
                        .slice_values(heights.min(), heights.min()),
                    min: heights.min(),
                    max: heights.max(),
                };
            },
            (edge, height) => get_or_die(this.edge_image_map, edge)
                .get(numeric_height(height)),
        ));
        if (start_preimage === undefined)
            throw new Error("Unreachable");
        let [start, height] = start_preimage
        return new Polygon(
            get_or_die(this.vertex_image_map, start)
                .get(numeric_height(height)),
            edges );
    }
}

class HeightedFaceGraph extends PlanarGraph {
    direction: Direction;
    edge_heights: Map<Edge, HeightRange>;
    face_heights: Map<Polygon, HeightRange>;
    iffy: {
        edges: Set<Edge>,
        faces: Set<Polygon>,
    };
    constructor(
        vertices: Iterable<Point>,
        edges: Iterable<Edge>,
        faces: Iterable<Polygon>,
        direction: Direction,
        max_height: number,
    ) {
        super(vertices, edges, faces);
        this.direction = direction;
        this.edge_heights = new Map()
        for (let edge of this.edges) {
            if (edge.direction === direction)
                continue;
            this.edge_heights.set( edge,
                {min: 0, max: max_height} );
            }
        this.face_heights = new Map()
        for (let face of this.faces) {
            this.face_heights.set( face,
                {min: 0, max: max_height} );
            }
        this.iffy = {
            edges: new Set(),
            faces: new Set(),
        }
    }

    get_edge_height(edge: Edge): HeightRange | null {
        let height = this.edge_heights.get(edge);
        if (height === undefined)
            return null;
        return height;
    }
    get_face_height(face: Polygon): HeightRange {
        let height = this.face_heights.get(face);
        if (height === undefined)
            throw new Error("Face does not belong");
        return height;
    }
    set_face_height(face: Polygon, h: number) {
        let height = this.get_face_height(face);
        if (height.height !== null) {
            throw new Error("Height already set");
        }
        if (h < height.min - EPSILON || h > height.max + EPSILON) {
            throw new Error("Impossible height");
        }
        height.height = height.max = height.min = h;
        this._mark_face_edges_iffy(face);
    }

    resolve_iffy_heights(): void {
        while (
            this.iffy.edges.size > 0 ||
            this.iffy.faces.size > 0
        ) {
            this._resolve_iffy_edges();
            this._resolve_iffy_faces();
        }
    }
    _mark_edge_faces_iffy(object: Edge) {
        for (let face of this.edge_faces(object)) {
            if (face === null)
                continue;
            this.iffy.faces.add(face);
        }
    }
    _mark_face_edges_iffy(object: Polygon) {
        for (let edge of object) {
            this.iffy.edges.add(edge);
        }
    }

    _resolve_iffy_edges(): void {
        for (let edge of this.iffy.edges) {
            let height = this.get_edge_height(edge),
                height_changed = false;
            if (height == null)
                continue;
            update_height: {
                if (height.height !== null)
                    break update_height;
                let [below_face, above_face] = this.edge_faces(edge);
                if (edge.delta.skew(this.direction) < 0) {
                    [below_face, above_face] = [above_face, below_face];
                }
                if (below_face !== null) {
                    let face_height = this.get_face_height(below_face);
                    if (face_height.min > height.min + EPSILON) {
                        height.min = face_height.min;
                        height_changed = true;
                    }
                }
                if (above_face !== null) {
                    let face_height = this.get_face_height(above_face);
                    if (face_height.max < height.max - EPSILON) {
                        height.max = face_height.max;
                        height_changed = true;
                    }
                }
                if (height_changed && (height.max - height.min < EPSILON)) {
                    height.height = height.max = height.min
                }
            }
            if (height_changed) {
                this._mark_edge_faces_iffy(edge);
            }
        }
        this.iffy.edges.clear();
    }
    _resolve_iffy_faces(): void {
        for (let face of this.iffy.faces) {
            let height = this.get_face_height(face),
                height_changed = false;
            update_height: {
                if (height.height !== null)
                    break update_height;
                let
                    below_edges: Array<[Edge,HeightRange]> = [],
                    above_edges: Array<[Edge,HeightRange]> = [];
                for (let {edge, vector} of face.oriented_edges()) {
                    let edge_height = this.get_edge_height(edge);
                    if (edge_height === null)
                        continue;
                    if (vector.skew(this.direction) > 0) {
                        above_edges.push([edge, edge_height]);
                    } else {
                        below_edges.push([edge, edge_height]);
                    }
                }
                for (let [edge, edge_height] of below_edges) {
                    if (edge_height.min > height.min + EPSILON) {
                        height.min = edge_height.min;
                        height_changed = true;
                    }
                }
                for (let [edge, edge_height] of above_edges) {
                    if (edge_height.max < height.max - EPSILON) {
                        height.max = edge_height.max;
                        height_changed = true;
                    }
                }
                if (height_changed && (height.max - height.min < EPSILON)) {
                    height.height = height.max = height.min
                }
            }
            if (height_changed) {
                this._mark_face_edges_iffy(face);
            }
        }
        this.iffy.faces.clear();
    }
}

class PRNG {
    // /**
    //  * @param numbers sorted array of at least two distinct numbers
    //  */
    // static add_number(numbers: Array<number>) {
    //     let ranges: Array<[number, number]> = [];
    //     for (let i = 1; i < numbers.length; ++i) {
    //         ranges.push([numbers[i-1], numbers[i]]);
    //     }
    // }
    static choose_element<T>(elements: Array<T>): T {
        return elements[PRNG.choose_index(elements.length)];
    }
    static choose_face(faces: Array<Polygon>): Polygon {
        return PRNG.choose_element(faces);
    }
    static choose_index(length: number) {
        let index = Math.floor(length * Math.random());
        if (index >= length)
            return index - 1;
        return index;
    }
}

