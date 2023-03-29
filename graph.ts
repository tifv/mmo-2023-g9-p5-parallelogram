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
    add(other: Vector) {
        return new Vector(this.x + other.x, this.y + other.y);
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
    static make_direction(vector: Vector) {
        let direction = new Direction(vector.x, vector.y);
        return DirectedVector.from_collinear(direction, vector);
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

