namespace Graphs {

class Pair {
    readonly x: number;
    readonly y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
    *[Symbol.iterator] () {
        yield this.x;
        yield this.y;
    }
}

export class AbstractVector extends Pair {
    opposite(): AbstractVector {
        return new AbstractVector(-this.x, -this.y);
    }
    unit(): UnitVector {
        return new UnitVector(this.x, this.y);
    }
    scale(scale: number): Vector {
        return new Vector(scale * this.x, scale * this.y);
    }
    project(this: Vector, other: UnitVector): number
    project(this: UnitVector, other: UnitVector): number
    project(direction: UnitVector) {
        return this.x * direction.x + this.y * direction.y;
    }
    dot(this: Vector, other: Vector): number
    dot(this: Vector, other: UnitVector): number
    dot(this: UnitVector, other: UnitVector): number
    dot(other: AbstractVector) {
        return this.x * other.x + this.y * other.y;
    }
    skew(this: Vector, other: Vector): number
    skew(this: Vector, other: UnitVector): number
    skew(this: UnitVector, other: UnitVector): number
    skew(other: AbstractVector): number {
        return this.y * other.x - this.x * other.y;
    }
}

export class UnitVector extends AbstractVector {
    get length(): 1 { return 1; }
    constructor(x: number, y: number) {
        let length = (x**2 + y**2)**(1/2);
        super(x / length, y / length);
    }
    opposite(): UnitVector {
        return new UnitVector(-this.x, -this.y);
    }
    perpendicular(): UnitVector {
        return new UnitVector(-this.y, this.x);
    }
}

const IS_DIRECTION = Symbol();

export class Direction extends UnitVector {
    get [IS_DIRECTION](): true { return true; }
    scale(scale: number) : DirectedVector {
        return new DirectedVector(this, scale);
    }
    static from_angle(angle: number) {
        return new Direction(Math.cos(angle), Math.sin(angle));
    }
}

export class Vector extends AbstractVector {
    static between(start: Point, end: Point) {
        return new Vector(end.x - start.x, end.y - start.y);
    }
    get length(): number {
        return (this.x**2 + this.y**2)**(1/2);
    }
    opposite(): Vector {
        return new Vector(-this.x, -this.y);
    }
    add(other: Vector) {
        return new Vector(this.x + other.x, this.y + other.y);
    }
    decompose(dir1: UnitVector, dir2: UnitVector): [number, number] {
        let det = dir1.skew(dir2);
        if (Math.abs(det) < EPSILON)
            throw new Error("Cannot decompose a vector " +
                "into collinear directions" );
        return [this.skew(dir2) / det, -this.skew(dir1) / det];
    }
}

export class DirectedVector extends Vector {
    readonly direction: Direction;
    readonly value: number;
    constructor(direction: Direction, value: number) {
        super(direction.x * value, direction.y * value)
        this.direction = direction;
        this.value = value;
    }
    scale(scale: number): DirectedVector {
        return new DirectedVector(this.direction, scale * this.value);
    }
    opposite(): DirectedVector {
        return new DirectedVector(this.direction, -this.value);
    }
    static from_collinear(
        direction: Direction, vector: Vector,
    ): DirectedVector {
        return new DirectedVector(direction, vector.project(direction));
    }
    static make_direction(vector: Vector) {
        let direction = new Direction(vector.x, vector.y);
        return DirectedVector.from_collinear(direction, vector);
    }
}

export class Point extends Pair {

    shift(vector: Vector): Point {
        return new Point(this.x + vector.x, this.y + vector.y);
    }
    is_equal(point: Point) {
        return ( Math.abs(this.x - point.x) < EPSILON &&
            Math.abs(this.y - point.y) < EPSILON );
    }

    static bbox(...points: Array<Point>): [Point, Point] {
        let
            min_x = +Infinity, max_x = -Infinity,
            min_y = +Infinity, max_y = -Infinity,
            n = points.length;
        if (n < 1)
            throw new Error("Cannot take bbox of an empty set");
        for (let {x, y} of points) {
            if (x < min_x - EPSILON) min_x = x;
            if (x > max_x + EPSILON) max_x = x;
            if (y < min_y - EPSILON) min_y = y;
            if (y > max_y + EPSILON) max_y = y;
        }
        return [new Point(min_x, min_y), new Point(max_x, max_y)];
    }

    static center(
        points: Array<Point>,
        weights?: Array<number>,
    ): Point {
        if (weights !== undefined && points.length !== weights.length)
            throw new Error("Weights must correspond to points");
        let total_weight = weights !== undefined
            ? weights.reduce((s, x) => s + x, 0)
            : points.length;
        if (Math.abs(total_weight) < EPSILON)
            throw new Error("Cannot take average of a zero-weight set");
        let x = 0, y = 0;
        for (let [index, point] of points.entries()) {
            let weight = weights !== undefined ? weights[index] : 1;
            x += point.x * weight;
            y += point.y * weight;
        }
        return new Point(x / total_weight, y / total_weight);
    }

    static incenter(pA: Point, pB: Point, pC: Point):
        {incenter: Point, inradius: number}
    {
        let incenter = Point.center( [pA, pB, pC], [
            Vector.between(pB, pC).length,
            Vector.between(pC, pA).length,
            Vector.between(pA, pB).length,
        ]);
        let sideAB = Vector.between(pA, pB);
        let inradius = Math.abs(Vector.between(pA, incenter).project(
            new UnitVector(sideAB.x, sideAB.y).perpendicular() ));
        return {incenter, inradius};
    }
}

export class Edge {
    readonly start: Point;
    readonly delta: DirectedVector;
    readonly end: Point;

    // id: any = crypto.randomUUID().substring(0, 8);

    constructor(start: Point, delta: DirectedVector, end: Point) {
        if (delta.value < 0)
            [start, delta, end] = [end, delta.opposite(), start];
        this.start = start;
        this.delta = delta;
        this.end = end;
        // Object.defineProperties(this, {
        //     start: {value: start, enumerable: true},
        //     delta: {value: delta, enumerable: true},
        //     end:   {value: end  , enumerable: true},
        // });
    }

    *[Symbol.iterator] (): IterableIterator<Point> {
        yield this.start;
        yield this.end;
    }
    incident_to(vertex: Point) {
        return vertex === this.start || vertex === this.end;
    }
    adjacent_to(edge: Edge) {
        return this.incident_to(edge.start) || this.incident_to(edge.end);
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

    project(point: Point): Point {
        let direction = this.direction;
        return point.shift(new UnitVector(-direction.y, direction.x).scale(
            Vector.between(point, this.start).skew(this.direction) ));
    }
    project_along(point: Point, direction: UnitVector): Point {
        return point.shift(direction.scale(
            Vector.between(point, this.start)
                .decompose(direction, this.direction)[0] ));
    }

    substitute(
        get_new_vertex: (vertex: Point) => Point,
    ): Edge {
        let
            start = get_new_vertex(this.start),
            end = get_new_vertex(this.end);
        if (start === this.start && end === this.end)
            return this;
        return new Edge(start, this.delta, end);
    }

    static *debug_edge_generation(edges: Generator<Edge,void,undefined>):
        Generator<Edge,void,undefined>
    {
        let previous_edge: Edge | null = null;
        for (let edge of edges) {
            if (previous_edge !== null && !edge.adjacent_to(previous_edge)) {
                let error: any = new Error("Emitted edges are not adjacent");
                error.info = {debug_edge_generation: {edge, previous_edge}};
                edges.throw(error);
                throw error;
            }
            previous_edge = edge;
            yield edge;
        }
    }
}

export type OrientedEdge = {
    edge: Edge, index: number,
    start: Point, end: Point,
    forward: boolean, vector: DirectedVector,
}

export class Polygon {
    edges: Array<Edge>;
    size: number;
    vertices: Array<Point>;
    parallelogram: ParallelogramInfo | null | undefined = undefined;

    // id: any = crypto.randomUUID().substring(0, 8);

    constructor(start: Point, edges: Array<Edge>) {
        this.edges = edges;
        this.size = edges.length;
        this.vertices = [];
        if (this.edges.length == 0) {
            this.vertices.push(start);
            return;
        }

        try {
            let vertex = start;
            for (let edge of this.edges) {
                this.vertices.push(vertex);
                vertex = edge.other_end(vertex);
            }
            if (vertex !== start)
                throw new Error("Polygon is not cyclic");
        } catch (error: any) {
            error.info = Object.assign({Polygon: {start, edges}}, error.info);
            throw error;
        }
    }

    get start(): Point {
        return this.vertices[0];
    }
    *[Symbol.iterator] (): IterableIterator<Edge> {
        yield* this.edges;
    }

    *oriented_edges(): Iterable<OrientedEdge> {
        let vertex = this.start;
        for (let [index, edge] of this.edges.entries()) {
            if (edge.start === vertex) {
                yield { edge, index,
                    start: edge.start, end: edge.end,
                    forward: true, vector: edge.delta };
            } else {
                yield { edge, index,
                    start: edge.end, end: edge.start,
                    forward: false, vector: edge.delta.opposite() };
            }
            vertex = edge.other_end(vertex);
        }
    }
    _index_modulo(index: number): number {
        return modulo(index, this.edges.length);
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
        return this._slice_cyclic_array( this.edges,
            start_index, end_index, {allow_empty} );
    }
    slice_oriented_edges(
        start_index: number,
        end_index: number,
        {allow_empty = false} = {},
    ): Array<OrientedEdge> {
        return this._slice_cyclic_array( Array.from(this.oriented_edges()),
            start_index, end_index, {allow_empty} );
    }
    _slice_cyclic_array<V>(
        array: Array<V>,
        start_index: number,
        end_index: number,
        {allow_empty = false} = {},
    ): Array<V> {
        start_index = this._index_modulo(start_index);
        end_index = this._index_modulo(end_index);
        if ( allow_empty && end_index >= start_index ||
            end_index > start_index )
        {
            return array.slice(start_index, end_index);
        }
        return array.slice(start_index).concat(
            array.slice(0, end_index) );
    }
    get_vertex(index: number): Point {
        if (this.edges.length == 0)
            return this.start;
        index = this._index_modulo(index);
        return this.vertices[index];
    }

    reversed(): Polygon {
        return new Polygon(this.start, Array.from(this.edges).reverse());
    }
    shift(vector: Vector): Polygon {
        let get_new_vertex: (vertex: Point) => Point; {
            let map = new Map<Point,Point>();
            for (let vertex of this.vertices) {
                map.set(vertex, vertex.shift(vector));
            }
            get_new_vertex = (vertex) => get_or_die(map, vertex);
        }
        return new Polygon( get_new_vertex(this.start),
            this.edges.map((edge) => edge.substitute(get_new_vertex)) );
    }

    substitute(
        get_new_vertex: (vertex: Point) => Point,
        get_new_edge: (edge: Edge) => (Edge | Edge[]),
    ): Polygon {
        let start = get_new_vertex(this.start);
        let edge_changes = false;
        let edges = [...function*(
            oriented_edges: Iterable<OrientedEdge>,
        ): Iterable<Edge> {
            for (let {edge, forward} of oriented_edges) {
                let replacement = get_new_edge(edge);
                if (replacement === edge) {
                    yield edge;
                    continue;
                }
                edge_changes = true;
                if (replacement instanceof Edge) {
                    yield replacement;
                    continue;
                }
                if (!forward) {
                    replacement = Array.from(replacement).reverse();
                }
                yield* replacement;
            }
        }(this.oriented_edges())];
        if (start === this.start && !edge_changes)
            return this;
        let polygon = new Polygon(start, edges);
        return polygon;
    }

    find_tangent_points(direction: UnitVector): {
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
        var lefts = Array.from(itermap(this.oriented_edges(), ({vector}) =>
            epsilon_sign(vector.skew(direction)) ));
        let
            forward_indices: Array<number> = [],
            backward_indices: Array<number> = [];
        for (let i = 0; i < this.edges.length; ++i) {
            let
                prev_left = lefts[this._index_modulo(i-1)],
                next_left = lefts[this._index_modulo(i)]
            if (prev_left <= 0 && next_left > 0) {
                forward_indices.push(i);
            }
            if (prev_left >= 0 && next_left < 0)
            {
                backward_indices.push(i);
            }
        }
        if (forward_indices.length < 1 || backward_indices.length < 1 ) {
            let error: any = new Error("Couldn't find a tangent");
            error.info = {find_tangent_points: {
                lefts, forward_indices, backward_indices,
                skews: itermap(this.oriented_edges(), ({vector}) =>
                    epsilon_sign(vector.skew(direction)) ),
            }};
            throw error;
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

    get_area(): number {
        let area = 0;
        let first_vertex = this.start;
        for (let {start, vector} of this.oriented_edges()) {
            area += vector.skew(Vector.between(first_vertex, start));
        }
        return area / 2;
    }
    contains(point: Point, allow_border: boolean = false): boolean {
        if (this.size < 2)
            return allow_border ? false : point.is_equal(this.start);
        const up = new UnitVector(1, 0);
        let vectors = this.vertices.map( vertex =>
            Vector.between(point, vertex) );
        type Relative = "left" | "right" | "up" | "down";
        let relative = vectors.map<Relative | null>((vector) => {
            let skew = vector.skew(up);
            if (skew > EPSILON)
                return "left";
            if (skew < EPSILON)
                return "right";
            let project = vector.project(up);
            if (project > EPSILON)
                return "up";
            if (project < EPSILON)
                return "down";
            return null;
        });
        let cycles: number = 0;
        for (let [index, vector] of vectors.entries()) {
            let
                prev_index = this._index_modulo(index-1),
                prev_vector = vectors[prev_index],
                skew = vector.skew(prev_vector),
                are_collinear = Math.abs(skew) < EPSILON,
                rel = relative[index], prev_rel = relative[prev_index];
            if (
                rel === null || prev_rel === null ||
                are_collinear && vector.dot(prev_vector) < EPSILON
            ) {
                return allow_border;
            }
            switch (rel) {
                case "left": switch (prev_rel) {
                    case "left":
                    case "down":
                        continue;
                    case "right":
                        if (skew < 0)
                            continue;
                        // fallthrough
                    case "up":
                        cycles += 1;
                        continue;
                    default:
                        throw new Error("unreachable")
                }
                case "right": switch (prev_rel) {
                    case "right":
                    case "up":
                    case "down":
                        continue;
                    case "left":
                        if (skew > 0)
                            continue;
                        cycles -= 1;
                        continue;
                    default:
                        throw new Error("unreachable")
                }
                case "up": switch (prev_rel) {
                    case "right":
                    case "up":
                        continue;
                    case "down":
                        return allow_border;
                    case "left":
                        cycles -= 1;
                        continue;
                    default:
                        throw new Error("unreachable")
                }
                case "down": switch (prev_rel) {
                    case "left":
                    case "right":
                    case "down":
                        continue;
                    case "up":
                        return allow_border;
                    default:
                        throw new Error("unreachable")
                }
            }
        }
        if (cycles === 1)
            return true;
        if (cycles !== 0)
            throw new Error("Polygon is not oriented correctly");
        return false;
    }

    static from_vectors(origin: Point, vectors: Array<DirectedVector>):
        Polygon
    {
        let last_point = origin;
        var edges: Array<Edge> = [];
        for (let vector of vectors) {
            let edge = Edge.from(last_point, vector);
            last_point = edge.other_end(last_point);
            edges.push(edge);
        }
        let n = edges.length;
        let last_edge = edges[n-1] =
            edges[n-1].replace_vertex_obj(last_point, origin);
        if (!last_edge.end.is_equal(
                last_edge.start.shift(last_edge.delta)
        )) {
            let error = new Error("Polygon has not cycled correctly");
            // @ts-ignore
            error.edges = edges;
            throw error;
        }
        return new Polygon(origin, edges);
    }

    static make_regular_even( center: Point,
        n: number, radius: number,
    ): {polygon: Polygon, directions: Direction[], side_length: number}
    {
        let directions: Array<Direction> = [];
        let vectors: Array<DirectedVector> = [];
        let centeral_angle = Math.PI/(n)
        let side_length = radius * 2 * Math.sin(centeral_angle/2);
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
            polygon: Polygon.from_vectors(
                center.shift(new Vector(
                    -side_length/2, radius * -Math.cos(centeral_angle/2) )),
                vectors
            ),
            directions: directions,
            side_length,
        };
    }

    get_sides(): Array<PolygonSide> {
        let sides = new Array<PolygonSide>();
        for (let {edge, forward, start} of this.oriented_edges()) {
            let direction = edge.direction;
            let side: PolygonSide | null = null;
            get_last_side: {
                if (sides.length === 0)
                    break get_last_side;
                let last_side = sides[sides.length-1];
                if (
                    last_side.direction !== direction ||
                    last_side.forward !== forward
                )
                    break get_last_side;
                side = last_side;
            }
            if (side === null) {
                side = {direction, forward, edges: [], start};
                sides.push(side);
            }
            side.edges.push(edge);
        }
        if (sides.length >= 2) {
            let first_side = sides[0], last_side = sides[sides.length-1];
            if (
                last_side.direction === first_side.direction &&
                last_side.forward   === first_side.forward
            ) {
                first_side.edges.unshift(...last_side.edges);
                sides.pop();
            }
        }
        return sides;
    }

    as_parallelogram(): Parallelogram | null {
        try_cache: {
            let polygon: (
                Polygon & {parallelogram: ParallelogramInfo} |
                Polygon & {parallelogram: null} |
                Polygon & {parallelogram: undefined}
            ) = this;
            if (polygon.parallelogram === undefined)
                break try_cache;
            if (polygon.parallelogram === null)
                return null;
            return polygon;
        }
        let sides = this.get_sides();
        let directions: Array<Direction> =
            Array.from(new Set(sides.map(({direction}) => direction)));
        if (directions.length > 2) {
            this.parallelogram = null;
            return null;
        }
        directions.sort((a, b) => a.skew(b));
        if (sides.length != 4) {
            this.parallelogram = null;
            return null;
        }
        let [index = null] = filtermap( sides, (run, index) =>
                (run.direction === directions[0] && run.forward === true) ?
                    index : null );
        if (index === null) {
            throw new Error("unreachable");
        }
        sides.splice(4 - index, 0, ...sides.splice(0, index));
        let [a, b, c, d] = sides;
        if (
            a.direction !== directions[0] || a.forward !== true  ||
            b.direction !== directions[1] || b.forward !== true  ||
            c.direction !== directions[0] || c.forward !== false ||
            d.direction !== directions[1] || d.forward !== false
        ) {
            // maybe the outer face, but not a true parallelogram;
            this.parallelogram = null;
            return null;
        }
        if (directions.length !== 2)
            throw new Error("unreachable");
        let info: ParallelogramInfo = {
            directions: <[Direction, Direction]>directions,
            sides: [a, b, c, d],
        };
        let polygon = Object.assign(this, {parallelogram: info});
        return polygon;
    }

    other_direction(this: Parallelogram, direction: Direction): Direction {
        for (let side of this.parallelogram.sides) {
            if (side.direction === direction)
                continue;
            return side.direction;
        }
        throw new Error("unreachable");
    }
}

type PolygonSide =
    {direction: Direction, forward: boolean, start: Point, edges: Edge[]}
type ParallelogramInfo = {
    /** Directions are ordered in the positive rotation order. */
    directions: [Direction, Direction],
    /** There is very specific order to the sides:
     *  - 1st side is the forward 1st direction;
     *  - 2nd side is the forward 2nd direction;
     *  - 3rd side is the backward 1st direction;
     *  - 4th side is the backward 2nd direction.
     *  This also must be the order in which the edges occur in
     *  the polygon itself.
     */
    sides: [
        PolygonSide,
        PolygonSide,
        PolygonSide,
        PolygonSide,
    ],
}
export type Parallelogram = Polygon & {parallelogram: ParallelogramInfo};

export type GraphLike = {
    vertices: Iterable<Point>,
    edges: Iterable<Edge>,
    faces: Iterable<Polygon>,
}

type EdgeSet = Array<Edge>;

/** First element is the left face (face with forward direction of the edge),
 *  second is the right face. */
type FaceSet = [Polygon | null, Polygon | null];

export class PlanarGraph implements GraphLike {
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
        this.faces = new Set(faces);
        try {
            this.edgemap = PlanarGraph._build_edgemap(this.edges);
            this.facemap = PlanarGraph._build_facemap(this.faces);
        } catch (error: any) {
            error.info = Object.assign({PlanarGraph: {
                graph: this,
            }}, error.info);
            throw error;
        }
    }
    static _build_edgemap(edges: Iterable<Edge>): PlanarGraph["edgemap"] {
        let edgemap: PlanarGraph["edgemap"] = new Map();
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
    get_vertex_edges(vertex: Point): EdgeSet {
        let edgeset = this.edgemap.get(vertex);
        if (edgeset === undefined)
            throw new Error("The vertex does not have an incident edge");
        return edgeset;
    }
    static _build_facemap(faces: Iterable<Polygon>): PlanarGraph["facemap"] {
        let facemap: PlanarGraph["facemap"] = new Map();
        for (let face of faces) {
            for (let {edge, forward} of face.oriented_edges()) {
                let faceset = facemap.get(edge);
                if (faceset === undefined) {
                    faceset = [null, null];
                    facemap.set(edge, faceset);
                }
                let index = forward ? 0 : 1;
                if (faceset[index] !== null) {
                    throw new Error(
                        "Face with the same direction is already present" );
                }
                faceset[index] = face;
            }
        }
        return facemap;
    }
    get_edge_faces(edge: Edge): FaceSet {
        let faceset = this.facemap.get(edge);
        if (faceset === undefined) {
            throw new Error(
                "The edge does not have an incident face" );
        }
        return faceset;
    }
    copy(): PlanarGraph {
        return new PlanarGraph(this.vertices, this.edges, this.faces);
    }
    substitute(
        get_new_vertex: (vertex: Point) => Point,
        get_new_edge: (edge: Edge) => (Edge | Edge[]),
    ): PlanarGraph {

        let vertices: Point[] = [];
        for (let vertex of this.vertices) {
            vertices.push(get_new_vertex(vertex));
        }

        let edges: Edge[] = [];
        for (let edge of this.edges) {
            let replacement = get_new_edge(edge);
            if (replacement instanceof Edge) {
                edges.push(replacement);
                continue;
            }
            let start = get_new_vertex(edge.start), vertex = start;
            let end = get_new_vertex(edge.end);
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
            faces.push(face.substitute(get_new_vertex, get_new_edge));
        }
        return new PlanarGraph(vertices, edges, faces);
    }

    check(options?: {[name: string]: boolean, return_errors: true}):
        {errors: any[], info: any[]}
    check(options?: {[name: string]: boolean}): void
    check({
        vertices_without_edges = true,
        vertices_with_rogue_edges = true,
        edges_with_rogue_vertices = true,
        edges_without_faces = true,
        edges_with_one_face = true,
        edges_with_rogue_faces = true,
        faces_with_rogue_edges = true,
        face_orientation = true,
        parallelograms = true,
        return_errors = false,
    }: {[name: string]: boolean} = {}): {errors: any[], info: any[]} | void {
        let errors: any[] = [], info: any[] = [{graph: this}];

        if (vertices_without_edges) {
            let lone_vertices = new Array<Point>();
            for (let vertex of this.vertices) {
                let edges = this.edgemap.get(vertex);
                if (edges === undefined) {
                    lone_vertices.push(vertex);
                    continue;
                }
            }
            if (lone_vertices.length > 0) {
                errors.push({vertices_without_edges: lone_vertices});
            }
        }
        if (vertices_with_rogue_edges) {
            let rogue_edges = new Array<Edge>();
            for (let vertex of this.vertices) {
                let edges = this.edgemap.get(vertex);
                if (edges === undefined)
                    continue;
                for (let edge of edges) {
                    if (this.edges.has(edge))
                        continue;
                    rogue_edges.push(edge);
                }
            }
            if (rogue_edges.length > 0) {
                errors.push({vertices_with_rogue_edges: rogue_edges});
            }
        }
        if (edges_with_rogue_vertices) {
            let rogue_vertices = new Map<Edge,Point[]>();
            for (let edge of this.edges) {
                let edge_rogue_vertices = new Array<Point>();
                for (let vertex of edge) {
                    if (!this.vertices.has(vertex)) {
                        edge_rogue_vertices.push(vertex);
                    }
                }
                if (edge_rogue_vertices.length > 0) {
                    rogue_vertices.set(edge, edge_rogue_vertices);
                }
            }
            if (rogue_vertices.size > 0) {
                errors.push({edges_with_rogue_vertices: rogue_vertices});
            }
        }

        if (edges_without_faces) {
            let zerosided_edges = new Array<Edge>();
            let onesided_edges = new Array<Edge>();
            for (let edge of this.edges) {
                let faces = this.facemap.get(edge);
                if (faces === undefined) {
                    zerosided_edges.push(edge);
                    continue;
                }
                if (!edges_with_one_face) {
                    continue;
                }
                if (faces[0] === null || faces[1] === null) {
                    onesided_edges.push(edge);
                    continue;
                }
            }
            if (zerosided_edges.length > 0 || onesided_edges.length > 0) {
                errors.push({edges_without_faces: Object.assign({},
                    zerosided_edges.length > 0 ? {zero: zerosided_edges} : null,
                    onesided_edges.length > 0 ? {one: onesided_edges} : null,
                    {all: zerosided_edges.concat(onesided_edges)},
                )});
            }
        }
        if (edges_with_rogue_faces) {
            let rogue_faces = new Array<Polygon>();
            for (let edge of this.edges) {
                let faces = this.facemap.get(edge);
                if (faces === undefined)
                    continue;
                for (let face of faces) {
                    if (face === null)
                        continue;
                    if (this.faces.has(face))
                        continue;
                    rogue_faces.push(face);
                }
            }
            if (rogue_faces.length > 0) {
                errors.push({edges_with_rogue_faces: rogue_faces});
            }
        }
        if (faces_with_rogue_edges) {
            let rogue_edges = new Map<Polygon,Edge[]>();
            for (let face of this.faces) {
                let face_rogue_edges = new Array<Edge>();
                for (let edge of face) {
                    if (!this.edges.has(edge)) {
                        face_rogue_edges.push(edge);
                    }
                }
                if (face_rogue_edges.length > 0) {
                    rogue_edges.set(face, face_rogue_edges);
                }
            }
            if (rogue_edges.size > 0) {
                errors.push({faces_with_rogue_edges: rogue_edges});
            }
        }

        if (face_orientation) {
            let reverse_faces = new Array<Polygon>();
            for (let face of this.faces) {
                if (face.get_area() < 0) {
                    reverse_faces.push(face);
                }
            }
            switch (reverse_faces.length) {
            case 0:
                errors.push({face_orientation: {reverse: []}});
                break;
            case 1:
                let [outer_face] = reverse_faces;
                info.push({outer_face});
                break;
            default:
                errors.push({face_orientation: {reverse: reverse_faces}});
                break;
            }
        }

        if (parallelograms) {
            let non_gram_faces = new Array<Polygon>();
            for (let face of this.faces) {
                let gram = face.as_parallelogram();
                if (gram === null) {
                    non_gram_faces.push(face)
                }
            }
            if (non_gram_faces.length <= 3) {
                info.push({parallelograms: {not: non_gram_faces}});
            } else {
                errors.push({parallelograms: {not: non_gram_faces}});
            }
        }

        if (return_errors)
            return {errors, info};
        if (errors.length > 0) {
            let error: any = new Error("Graph integrity compromised");
            error.info = { Graph_check:
                Object.assign({}, ...info, ...errors) };
            throw error;
        }
    }

    _remove_face(face: Polygon): void {
        let deleted: boolean = this.faces.delete(face);
        if (!deleted)
            throw new Error("The face did not belong to the graph");
        let dangling_edges = new Array<Edge>();
        for (let {edge, forward} of face.oriented_edges()) {
            let faceset = this.get_edge_faces(edge);
            let index = forward ? 0 : 1;
            if (faceset[index] !== face)
                throw new Error( "The incidence of face with the edge " +
                    "was not registered" );
            faceset[index] = null;
            if (faceset[0] === null && faceset[1] === null)
                dangling_edges.push(edge);
        }
        for (let edge of dangling_edges) {
            this._remove_edge(edge);
        }
    }
    _remove_edge(edge: Edge): void {
        let faceset = this.facemap.get(edge);
        if (faceset === undefined) {
            throw new Error("The edge incidences were not registered");
        }
        if (faceset[0] !== null || faceset[1] !== null) {
            throw new Error("The edge incidences were not removed");
        }
        this.facemap.delete(edge);
        let deleted = this.edges.delete(edge);
        if (!deleted)
            throw new Error("The edge did not belong to the graph");
        let dangling_vertices = new Array<Point>();
        for (let vertex of edge) {
            let edgeset = this.get_vertex_edges(vertex);
            let index = edgeset.indexOf(edge);
            if (index < 0)
                throw new Error( "The incidence of edge with the vertex " +
                    "was not registered" );
            edgeset.splice(index, 1);
            if (edgeset.length === 0)
                dangling_vertices.push(vertex);
        }
        for (let vertex of dangling_vertices) {
            this._remove_vertex(vertex);
        }
    }
    _remove_vertex(vertex: Point): void {
        let edgeset = this.edgemap.get(vertex);
        if (edgeset === undefined) {
            throw new Error("The vertex incidences were not registered");
        }
        if (edgeset.length > 0) {
            throw new Error("The vertex incidences were not removed");
        }
        this.edgemap.delete(vertex);
        let deleted = this.vertices.delete(vertex);
        if (!deleted)
            throw new Error("The vertex did not belong to the graph");
    }
    _add_face(face: Polygon): void {
        if (this.faces.has(face))
            throw new Error("The face already belongs to the graph");
        this.faces.add(face);
        for (let {edge, forward} of face.oriented_edges()) {
            if (!this.edges.has(edge))
                this._add_edge(edge);
            let faceset = get_or_die(this.facemap, edge);
            let index = forward ? 0 : 1;
            if (faceset[index] !== null) {
                throw new Error(
                    "Face with the same direction is already present" );
            }
            faceset[index] = face;
        }
    }
    _add_edge(edge: Edge): void {
        if (this.edges.has(edge))
            throw new Error("The edge already belongs to the graph");
        this.edges.add(edge);
        for (let vertex of edge) {
            if (!this.vertices.has(vertex))
                this._add_vertex(vertex);
            let edgeset = get_or_die(this.edgemap, vertex);
            if (edgeset.includes(edge)) {
                throw new Error("unreachable");
            }
            edgeset.push(edge);
        }
        this.facemap.set(edge, [null, null]);
    }
    _add_vertex(vertex: Point): void {
        if (this.vertices.has(vertex))
            throw new Error("The vertex already belongs to the graph");
        this.vertices.add(vertex);
        this.edgemap.set(vertex, []);
    }
    _join_edges( vertex: Point,
        special_faces: Record<string,Polygon>,
    ): Edge {
        let edgeset = this.get_vertex_edges(vertex);
        if (edgeset.length !== 2) {
            throw new Error("Can only join two adjacent edges");
        }
        let [edge1, edge2] = edgeset;
        let direction = edge1.direction;
        if (
            direction !== edge2.direction ||
            epsilon_sign(edge1.delta.value) !==
                epsilon_sign(edge2.delta.value)
        ) {
            throw new Error("Can only join two co-directed edges");
        }
        if (edge1.start === vertex) {
            [edge1, edge2] = [edge2, edge1];
        }
        if (edge1.end !== vertex || edge2.start !== vertex) {
            throw new Error("unreachable");
        }
        let start = edge1.start, end = edge2.end;
        let
            faceset1 = this.get_edge_faces(edge1),
            faceset2 = this.get_edge_faces(edge2);
        if (faceset1[0] !== faceset2[0] || faceset1[1] !== faceset2[1]) {
            throw new Error("unreachable");
        }
        let new_edge = new Edge(
            start,
            new DirectedVector( direction,
                edge1.delta.value + edge2.delta.value ),
            end,
        );
        let new_faces = new Array<Polygon>();
        let [face1, face2] = faceset1;
        if (face1 === null && face2 === null)
            throw new Error("unreachable");
        if (face1 !== null) {
            let index1 = face1.edges.indexOf(edge1);
            if (index1 < 0)
                throw new Error("unreachable");
            if (face1.get_edge(index1+1) !== edge2)
                throw new Error("unreachable");
            let new_face1 = new Polygon(
                start,
                [new_edge, ...face1.slice_edges(index1+2, index1)],
            );
            new_faces.push(new_face1);
            this._remove_face(face1);
            record_substitute(special_faces, face1, new_face1);
        }
        if (face2 !== null) {
            let index2 = face2.edges.indexOf(edge2);
            if (index2 < 0)
                throw new Error("unreachable");
            if (face2.get_edge(index2+1) !== edge1)
                throw new Error("unreachable");
            let new_face2 = new Polygon(
                end,
                [new_edge, ...face2.slice_edges(index2+2, index2)],
            );
            new_faces.push(new_face2);
            this._remove_face(face2);
            record_substitute(special_faces, face2, new_face2);
        }
        if (this.edges.has(edge1) || this.edges.has(edge2))
            throw new Error("unreachable");
        for (let face of new_faces) {
            this._add_face(face);
        }
        return new_edge;
    }
    _join_parallelograms( edge: Edge,
        special_faces: Record<string,Polygon>,
    ): Polygon {
        let [face1, face2] = this.get_edge_faces(edge);
        if (face1 === null || face2 === null)
            throw new Error("Can only join two adjacent faces");
        let
            gram1 = face1.as_parallelogram(),
            gram2 = face2.as_parallelogram();
        if (gram1 === null || gram2 === null)
            throw new Error("Can only join two adjacent parallelograms");
        let direction = edge.direction;
        let start = edge.start;
        let other_direction = gram1.other_direction(direction);
        if (gram2.other_direction(direction) !== other_direction) {
            throw new Error( "Can only join two parallelograms " +
                "with parallel respective edges" );
        }
        let
            [a1, b1, c1, d1] = gram1.parallelogram.sides,
            [a2, b2, c2, d2] = gram2.parallelogram.sides;
        let edges: Array<Edge>;
        if (other_direction.skew(direction) > 0) {
            if (
                a1.edges.length > 1 || a1.edges[0] !== edge ||
                c2.edges.length > 1 || c2.edges[0] !== edge
            )
                throw new Error( "Can only join parallelograms " +
                    "with single common edge" );
            edges = [
                ...d2.edges, ...a2.edges, ...b2.edges,
                ...b1.edges, ...c1.edges, ...d1.edges,
            ];
        } else {
            if (
                b1.edges.length > 1 || b1.edges[0] !== edge ||
                d2.edges.length > 1 || d2.edges[0] !== edge
            )
                throw new Error( "Can only join parallelograms " +
                    "with single common edge" );
            edges = [
                ...a2.edges, ...b2.edges, ...c2.edges,
                ...c1.edges, ...d1.edges, ...a1.edges,
            ];
        }
        let new_face = new Polygon(start, edges);
        this._remove_face(face1);
        this._remove_face(face2);
        record_substitute(special_faces, face1, null);
        record_substitute(special_faces, face2, null);
        this._add_face(new_face);
        return new_face;
    }
    reduce_parallelograms<F extends string>(
        special_faces: Record<F,Polygon>,
    ): Record<F,Polygon> {
        let iffy_vertices = new Set<Point>(this.vertices);
        let iffy_edges = new Set<Edge>(this.edges);
        while (iffy_vertices.size > 0 || iffy_edges.size > 0) {
            for (let vertex of iffy_vertices) {
                let edgeset = this.get_vertex_edges(vertex);
                if (edgeset.length !== 2)
                    continue;
                let [edge1, edge2] = edgeset;
                if (edge1.direction !== edge2.direction)
                    continue;
                let new_edge = this._join_edges(vertex, special_faces);
                iffy_edges.delete(edge1);
                iffy_edges.delete(edge2);
                iffy_edges.add(new_edge);
            }
            iffy_vertices.clear();
            for (let edge of iffy_edges) {
                let [face1, face2] = this.get_edge_faces(edge);
                if (face1 === null || face2 === null)
                    continue;
                let
                    gram1 = face1.as_parallelogram(),
                    gram2 = face2.as_parallelogram();
                if (gram1 === null || gram2 === null)
                    continue;
                let direction = edge.direction;
                let other_direction = gram1.other_direction(direction);
                if (gram2.other_direction(direction) !== other_direction)
                    continue;
                let side1: PolygonSide, side2: PolygonSide;
                if (other_direction.skew(direction) > 0) {
                    side1 = gram1.parallelogram.sides[0];
                    side2 = gram2.parallelogram.sides[2];
                } else {
                    side1 = gram1.parallelogram.sides[1];
                    side2 = gram2.parallelogram.sides[3];
                }
                if (side1.edges.length > 1 || side2.edges.length > 1)
                    continue;
                this._join_parallelograms(edge, special_faces);
                for (let vertex of edge)
                    iffy_vertices.add(vertex);
            }
            iffy_edges.clear();
        }
        return special_faces;
    }
    /**
     *  @yields series of points on edges
     *  intermitted with corresponding faces
     */
    *trace_through_parallelograms({
        start,
        direction,
        forward,
    }: {
        start: {
            point: Point,
            face: Parallelogram,
        },
        direction: Direction,
        forward: boolean,
    }):
        Iterable<Point|Parallelogram>
    {
        let point: Point, face: Parallelogram | null;
        ({point, face} = start);
        while (true) {
            yield point;
            if (face === null)
                break;
            yield face;
            let trace_direction: UnitVector = face.other_direction(direction);
            if ((forward ? +1 : -1) * trace_direction.skew(direction) > 0)
                trace_direction = trace_direction.opposite();
            let side: PolygonSide | null = null;
            for (let s of face.parallelogram.sides) {
                if (s.direction === direction && s.forward === forward) {
                    side = s;
                    break;
                }
            }
            if (side === null)
                throw new Error("Direction does not belong to the face");
            let edge: Edge | null = null;
            for (let e of side.edges) {
                let end = forward ? e.end : e.start;
                if (Vector.between(point, end).skew(trace_direction)
                        > -EPSILON
                ) {
                    edge = e;
                    break;
                }
            }
            if (edge === null) {
                edge = side.edges[side.edges.length - 1];
            }
            point = edge.project_along(point, trace_direction);
            let [other_face] = this.get_edge_faces(edge)
                .filter(f => f !== face);
            if (other_face === null) {
                face = null;
                continue;
            }
            face = other_face.as_parallelogram();
        }
    }
}

function record_substitute<V>(obj: Record<string,V>, orig: V, repl: V) {
    for (let [key, value] of Object.entries(obj)) {
        if (value === orig)
            obj[key] = repl;
    }
}

}

import Vector = Graphs.Vector;
import Direction = Graphs.Direction;
import DirectedVector = Graphs.DirectedVector;
import Point = Graphs.Point;
import Edge = Graphs.Edge;
import Polygon = Graphs.Polygon;
import Parallelogram = Graphs.Parallelogram;
import PlanarGraph = Graphs.PlanarGraph;

