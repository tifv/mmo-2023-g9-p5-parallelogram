namespace Graphs {

class Pair {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
        // Object.defineProperties(this, {
        //     x: {value: x, enumerable: true},
        //     y: {value: y, enumerable: true},
        // });
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
    project(direction: Direction) {
        return this.x * direction.x + this.y * direction.y;
    }
    skew(vector: AbstractVector) {
        return this.y * vector.x - this.x * vector.y;
    }
}

export class Direction extends AbstractVector {
    constructor(x: number, y: number) {
        let length = (x**2 + y**2)**(1/2);
        super(x / length, y / length);
    }
    static from_angle(angle: number) {
        return new Direction(Math.cos(angle), Math.sin(angle));
    }
}

export class Vector extends AbstractVector {
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

export class DirectedVector extends Vector {
    direction: Direction;
    value: number;
    constructor(direction: Direction, value: number) {
        super(direction.x * value, direction.y * value)
        this.direction = direction;
        this.value = value;
        // Object.defineProperties(this, {
        //     direction: {value: direction, enumerable: true},
        //     value:     {value: value    , enumerable: true},
        // });
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
}

export class Edge {
    start: Point;
    delta: DirectedVector;
    end: Point;

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

    *[Symbol.iterator] (): Generator<Point, void, undefined> {
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

    *[Symbol.iterator] (): Generator<Edge, void, undefined> {
        yield* this.edges;
    }
    *oriented_edges(): Generator<OrientedEdge, void, undefined> {
        let vertex = this.vertices[0];
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
        // set to true to ensure start of the generator in case of refactoring
        let edge_changes = true;
        function* new_edges(
            oriented_edges: Generator<OrientedEdge, void, undefined>,
        ): Generator<Edge, void, undefined> {
            edge_changes = false;
            for (let {edge, forward} of oriented_edges) {
                let replacement = edge_map(edge);
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
        }
        let start = vertex_map(this.vertices[0]);
        let edges = Array.from(new_edges(this.oriented_edges()));
        if (start === this.vertices[0] && !edge_changes)
            return this;
        let polygon = new Polygon(start, edges);
        return polygon;
    }

    find_tangent_points(direction: AbstractVector): {
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
        let first_vertex = this.vertices[0];
        for (let {start, vector} of this.oriented_edges()) {
            area += vector.skew(Vector.from_points(first_vertex, start));
        }
        return area / 2;
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

}

export type GraphLike = {
    vertices: Iterable<Point>,
    edges: Iterable<Edge>,
    faces: Iterable<Polygon>,
}

type EdgeSet = Array<Edge>;

/** First element is the left face (face with forward direction of the edge),
 *  second is the right face. */
type FaceSet = [Polygon | null, Polygon | null];

export class PlanarGraph {
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
            throw new Error("The vertex does not have an incident edge");
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
                let index = forward ? 0 : 1;
                if (faceset[index] !== null) {
                    let error: any = new Error(
                        "Face with the same direction is already present" );
                    error.info = {build_facemap:
                        {edge, face, faceset} };
                    throw error;
                }
                faceset[index] = face;
            }
        }
        return facemap;
    }
    edge_faces(edge: Edge): FaceSet {
        let faceset = this.facemap.get(edge);
        if (faceset === undefined) {
            let error: any = new Error(
                "The edge does not have an incident face" );
            error.info = {edge_faces: {edge, graph: this}}
            throw error;
        }
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

    check(options?: {[name: string]: boolean, return_errors: true}):
        {errors: any[], info: any[]}
    check(options?: {[name: string]: boolean}): void
    check({
        vertices_without_edges = true,
        edges_with_rogue_vertices = true,
        edges_without_faces = true,
        edges_with_one_face = true,
        faces_with_rogue_edges = true,
        face_orientation = true,
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

        face_orientation:
        if (face_orientation) {
            let reverse_faces = new Array<Polygon>();
            for (let face of this.faces) {
                if (face.get_area() < 0) {
                    reverse_faces.push(face);
                }
            }
            if (reverse_faces.length == 0) {
                errors.push({face_orientation: {reverse: []}});
                break face_orientation;
            }
            if (reverse_faces.length == 1) {
                let [outer_face] = reverse_faces;
                info.push({outer_face})
                break face_orientation;
            }
            errors.push({face_orientation: {reverse: reverse_faces}});
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
}

}

import Direction = Graphs.Direction;
import DirectedVector = Graphs.DirectedVector;
import Point = Graphs.Point;
import Edge = Graphs.Edge;
import Polygon = Graphs.Polygon;
import PlanarGraph = Graphs.PlanarGraph;

