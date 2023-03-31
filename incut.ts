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

function construct_cut_region(uncut_region: UncutRegion, flows: Flows) {
    let origin = uncut_region.polygon.vertices[0];
    let [vec1, vec2, vec3] = Array.from(
        uncut_region.triangle1.oriented_edges()
    ).map(({vector}) => vector);
    let region = CutRegion.initial(origin, [vec1, vec2, vec3]);
    var
        sector_start = origin,
        sector_end = region.triangle2.vertices[0];
    // XXX debug
    var region_history: any[] = [];
    region_history.push({region, sector_start, sector_end});
    let set_edge_generation = () => {
        region.graph.edges.forEach((edge) => {
            if (edge.id.generation === undefined)
                edge.id.generation = region_history.length;
        });
    };
    set_edge_generation();
    for (let [direction, flow] of Chooser.choose_order(Array.from(flows))) {
        try {
            ({region, sector_start, sector_end} =
                Incutter.incut(
                    region, sector_start, sector_end,
                    direction, flow,
                ));
            set_edge_generation();
            region_history.push({region, sector_start, sector_end});
            region.graph.check();
        } catch (error: any) {
            error.info = Object.assign({construct_cut_region: {
                region_history,
            }}, error.info);
            throw error
        }
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
    // heighted_graph: HeightedFaceGraph;
    edge_height_map: Map<Edge, NumberSet> = new Map();
    vertex_height_map: Map<Point, NumberSet> = new Map();

    vertex_groups: Set<Point|VertexGroup> = new Set();

    vertex_image_map: Map<Point,NumberMap<Point>> = new Map();
    new_edge_map: Map<Point,SlicedArray<Edge>> = new Map();
    edge_image_map: Map<Edge,NumberMap<Edge|Edge[]>> = new Map();
    new_face_map: Map<Edge,Polygon[]> = new Map();
    face_image_map: Map<Polygon,Polygon> = new Map();

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
        incutter.determine_face_heights(heighted_graph, heights);
        heighted_graph.check();
        incutter.determine_edge_heights(heighted_graph);
        incutter.determine_vertex_heights();

        let new_outer_face: Polygon;
        let new_graph: PlanarGraph;
        try {
            incutter.group_vertices();
            incutter.generate_vertex_images();
            incutter.generate_edge_images();
            incutter.generate_face_images(heighted_graph);
            new_outer_face = incutter.regenerate_polygon().reversed();
            new_graph = new PlanarGraph(
                [...incutter.all_new_vertices()],
                [...incutter.all_new_edges()],
                [...incutter.all_new_faces(), new_outer_face],
            );
        } catch (error: any) {
            error.info = Object.assign({incut: {
                heighted_graph, incutter,
            }}, error.info);
            throw error;
        }
        return {
            region: new CutRegion(
                new_graph,
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
        } = this.polygon.find_tangent_points(direction));
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
        return new HeightedFaceGraph(
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
    }
    determine_face_heights( heighted_graph: HeightedFaceGraph,
        heights: number[],
    ) {
        while (true) {
            heighted_graph.resolve_iffy_heights();
            let floating_faces = new Array<Polygon>();
            for (let [face, height] of heighted_graph.face_heights) {
                if (height.height === undefined)
                    floating_faces.push(face);
            }
            if (floating_faces.length == 0)
                break;
            let
                floating_face = Chooser.choose_face(floating_faces),
                height = heighted_graph.get_face_height(floating_face);
            // XXX set any possible intermediate heights, not just min and max
            heighted_graph.set_face_height( floating_face,
                Chooser.choose_element([height.min, height.max]) );
        }
        /**
         * XXX TODO add any underused intermediate heights to vertices
         * and connect them to other objects that use them.
         * (Track usage of each height in operations above.
         * Or the number of occurences.)
         */
    }
    determine_edge_heights(heighted_graph: HeightedFaceGraph) {
        for (let edge of heighted_graph.edges) {
            let edge_heights: NumberSet = new NumberSet();
            for (let face of heighted_graph.edge_faces(edge)) {
                if (face === null)
                    continue;
                let face_height =
                    heighted_graph.get_face_height(face).height;
                if (face_height === undefined)
                    throw new Error("Not all face heights were defined");
                edge_heights.add(face_height);
            }
            this.edge_height_map.set(edge, edge_heights);
            // XXX debug
            if (edge.direction !== this.direction) {
                let {below: below_face, above: above_face}
                    = heighted_graph.relative_edge_faces(edge);
                let [below_height, above_height] = [below_face, above_face]
                    .map( face => (face == null)
                        ? null : heighted_graph.get_face_height(face).height );
                if ( below_height != null && above_height != null &&
                    below_height > above_height )
                {
                    let error: any = new Error("Face heights are not ordered");
                    error.info = {determine_edge_heights: {
                        below_face, above_face, edge,
                    }}
                }
            }
        }
        for (let edge of this.lower_border_edges) {
            get_or_die(this.edge_height_map, edge).add(0);
        }
        for (let edge of this.upper_border_edges) {
            get_or_die(this.edge_height_map, edge).add(this.max_height);
        }
    }
    determine_vertex_heights() {
        for (let vertex of this.graph.vertices) {
            let heights: NumberSet = new NumberSet();
            for (let edge of this.graph.vertex_edges(vertex)) {
                let edge_heights = get_or_die(this.edge_height_map, edge);
                for (let height of edge_heights)
                    heights.add(height);
            }
            this.vertex_height_map.set(vertex, heights);
        }
    }
    group_vertices() {
        let group_builder = new Map<Point,VertexGroup>();
        for (let edge of this.graph.edges) {
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
        for (let vertex of this.graph.vertices) {
            let vertex_group = group_builder.get(vertex);
            if (vertex_group !== undefined) {
                this.vertex_groups.add(vertex_group);
            } else {
                this.vertex_groups.add(vertex);
            }
        }
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
        new_edges.insert_guard(projection_images.min_key());
        for (let i = 1; i < projection_images.length; ++i) {
            let
                [start_proj, start] = projection_images[i-1],
                [end_proj, end] = projection_images[i];
            new_edges.insert_slice(start_proj, [new Edge(
                start,
                DirectedVector.from_collinear( this.direction,
                    Vector.from_points(start, end) ),
                end,
            )], end_proj);
        }
        { // XXX debug
            let projection_span =
                projection_images.max_key() - projection_images.min_key();
            let new_edges_span = Array.from(new_edges.slice_values()).reduce(
                (prev: number, edge) =>
                    prev + edge.delta.project(this.direction),
                0 );
            if (
                projection_images.max_key() != new_edges.max_key() ||
                projection_images.min_key() != new_edges.min_key() ||
                Math.abs(projection_span - new_edges_span) > EPSILON
            ) {
                let error: any = new Error(
                    "Vertex images are inconsistent with edges between them" );
                error.info = {vertex_group: {vertices, edges}};
                throw error;
            }
        }

        for (let [vertex, v_projections] of vertex_data) {
            let v_edge_map = new SlicedArray<Edge>();
            v_edge_map.insert_guard(v_projections.min_key());
            for (let i = 1; i < v_projections.length; ++i) {
                let [height, projection] = v_projections[i-1];
                let [next_height, next_proj] = v_projections[i];
                v_edge_map.insert_slice( height,
                    new_edges.slice_values(projection, next_proj),
                next_height );
            }
            this.new_edge_map.set(vertex, v_edge_map);
        }

        for (let edge of edges) {
            let heights = get_or_die(this.edge_height_map, edge);
            let edge_images = new NumberMap<Edge[]>();
            let [start_images, end_images] = [edge.start, edge.end]
                .map((vertex) => get_or_die(this.vertex_image_map, vertex));
            for (let height of heights) {
                let [start_proj, end_proj] = [start_images, end_images]
                    .map((image_map) => image_map.get(height))
                    .map((image) => projection_images.find_value(image).key);
                edge_images.set(height,
                    new_edges.slice_values(start_proj, end_proj) );
            }
            this.edge_image_map.set(edge, edge_images);

            // XXX debug
            if ( edge_images.some(([,edges]) =>
                edges.some((edge) => edge.id.generation !== undefined))
            ) {
                let error: any = new Error("Edge has old images");
                error.info = {vertex_group: {vertices, edges},
                    old_edges: {edge, edge_images} };
                throw error;
            }
        }

    }
    * all_new_vertices(): Generator<Point,void,undefined> {
        for (let [, images] of this.vertex_image_map) {
            for (let [, image] of images) {
                yield image;
            }
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
            // XXX debug
            if ( edge_images.some( ([,edge]) =>
                edge.id.generation !== undefined )
            ) {
                let error: any = new Error("Edge has old images");
                error.info = {old_edges: {edge, edge_images}};
                throw error;
            }

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
    * all_new_edges(): Generator<Edge,void,undefined> {
        for (let [, images] of this.edge_image_map) {
            for (let [, image] of images) {
                if (image instanceof Edge) {
                    yield image;
                } else {
                    yield* image;
                }
            }
        }
        for (let [, edges] of this.new_edge_map) {
            yield* edges.slice_values();
        }
    }
    generate_face_images(heighted_graph: HeightedFaceGraph) {
        for (let [face, {height: maybe_height}]
            of heighted_graph.face_heights
        ) {
            if (maybe_height === undefined) {
                throw new Error("Not all polygons have heights.")
            }
            let height = maybe_height;
            let new_face = face.substitute(
                (vertex) =>
                    get_or_die(this.vertex_image_map, vertex).get(height),
                (edge) =>
                    get_or_die(this.edge_image_map, edge).get(height),
            );
            this.face_image_map.set(face, new_face);

            // XXX debug
            if (new_face.edges.some(edge => (edge.id.generation !== undefined))) {
                let error: any = new Error("Face has old edges");
                error.info = {face_images: {face, new_face}};
                throw error;
            }
        }
    }
    * all_new_faces(): Generator<Polygon,void,undefined> {
        for (let [, image] of this.face_image_map) {
            yield image;
        }
        for (let [, faces] of this.new_face_map) {
            yield* faces;
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
            // XXX debug
            let image_distances = new Array<any>();
            for ( let {edge, index, forward, vector}
                of polygon.oriented_edges() )
            {
                let
                    start = forward ? edge.start : edge.end,
                    end = edge.other_end(start);
                if (prev_vector === null) {
                    let prev_edge = polygon.get_edge(index-1);
                    prev_vector = prev_edge.delta_from(
                        prev_edge.other_end(start) );
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
                    for (let e of start_image) {
                        try {
                            image_distances.push([
                                Vector.from_points(start, e.start).length,
                                Vector.from_points(start, e.end).length ]);
                            yield e;
                        } catch (error: any) {
                            error.info = Object.assign(
                                {edge_generator: {case: 1, edge: e}},
                                error.info );
                            throw error;
                        }
                    }
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
                    try {
                        image_distances.push(
                            Vector.from_points(edge.start, edge_image.start).length );
                        yield edge_image;
                    } catch (error: any) {
                        error.info = Object.assign(
                            {edge_generator: { case: 2,
                                edge, edge_image, edge_height, start_height,
                                image_distances
                            }},
                            error.info );
                        throw error;
                    }
                } else {
                    if (!forward) {
                        edge_image = Array.from(edge_image).reverse();
                    }
                    for (let e of edge_image) {
                        try {
                            image_distances.push(
                                Vector.from_points(edge.start, e.start).length );
                            yield e;
                        } catch (error) {
                            throw error;
                        }
                    }
                }
                prev_vector = vector;
            }
        }
        // XXX debug
        let edges = Array.from(Edge.debug_edge_generation(edge_generator(
            this.polygon,
            (vertex) => {
                let heights = get_or_die(this.vertex_height_map, vertex);
                // XXX debug
                let edges: SlicedArray<Edge> | null = null;
                try {
                    edges = get_or_die(this.new_edge_map, vertex);
                    return {
                        edges: get_or_die(this.new_edge_map, vertex)
                            .slice_values(heights.min(), heights.max()),
                        min: heights.min(),
                        max: heights.max(),
                    };
                } catch (error: any) {
                    error.info = Object.assign({vertex_edge_map: {vertex, heights, edges}})
                    throw error;
                }
            },
            (edge, height) => get_or_die(this.edge_image_map, edge)
                .get(numeric_height(height)),
        )));
        if (start_preimage === undefined)
            throw new Error("Unreachable");
        let [start, height] = start_preimage;
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

    // XXX debug
    iffy_history: {
        edges: Array<[Edge, Polygon, boolean, number, number]>,
        faces: Array<[Polygon, Edge, boolean, number, number]>,
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
        // XXX debug
        this.iffy_history = {
            edges: new Array<[Edge,Polygon,boolean,number,number]>(),
            faces: new Array<[Polygon,Edge,boolean,number,number]>(),
        };
    }

    relative_edge_faces(edge: Edge):
        {above: Polygon | null, below: Polygon | null}
    {
        if (edge.direction == this.direction)
            throw new Error("This is a vertical edge");
        let [below_face, above_face] = this.edge_faces(edge);
        if (edge.direction.skew(this.direction) < 0) {
            [below_face, above_face] = [above_face, below_face];
        }
        return {below: below_face, above: above_face};
    }
    relative_face_edges(face: Polygon):
        {above: Edge[], below: Edge[]}
    {
        let
            below_edges = new Array<Edge>(),
            above_edges = new Array<Edge>();
        for (let {edge, vector} of face.oriented_edges()) {
            if (edge.direction === this.direction)
                continue;
            if (vector.skew(this.direction) > 0) {
                above_edges.push(edge);
            } else {
                below_edges.push(edge);
            }
        }
        return {below: below_edges, above: above_edges};
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
        if (height.height !== undefined) {
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
    _mark_edge_faces_iffy(edge: Edge) {
        for (let face of this.edge_faces(edge)) {
            if (face === null)
                continue;
            this.iffy.faces.add(face);
        }
    }
    _mark_face_edges_iffy(face: Polygon) {
        for (let edge of face) {
            this.iffy.edges.add(edge);
        }
    }

    _resolve_iffy_edges(): void {
        for (let edge of this.iffy.edges) {
            if (edge.direction === this.direction)
                continue;
            let height = this.get_edge_height(edge);                
            if (height == null)
                throw new Error("unreachable");
            let height_changed = false;
            update_height: {
                if (height.height !== undefined)
                    break update_height;
                let {below: below_face, above: above_face}
                    = this.relative_edge_faces(edge);
                if (below_face !== null) {
                    let face_height = this.get_face_height(below_face);
                    if (face_height.min > height.min + EPSILON) {
                        height.min = face_height.min;
                        height_changed = true;
                    }
                    this.iffy_history.edges.push(
                        [edge, below_face, true, height.min, face_height.min] );
                }
                if (above_face !== null) {
                    let face_height = this.get_face_height(above_face);
                    if (face_height.max < height.max - EPSILON) {
                        height.max = face_height.max;
                        height_changed = true;
                    }
                    this.iffy_history.edges.push(
                        [edge, above_face, false, height.max, face_height.max] );
                }
                if (height_changed && (height.max - height.min < EPSILON)) {
                    if (height.max - height.min < -EPSILON) {
                        throw this._inordered_heights_error({edge});
                    }
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
            let height = this.get_face_height(face);
            let height_changed = false;
            update_height: {
                if (height.height !== undefined)
                    break update_height;
                let {below: below_edges, above: above_edges} =
                    this.relative_face_edges(face);
                let [below_heights, above_heights] =
                    [below_edges, above_edges].map(edges => edges.map(edge => {
                        let edge_height = this.get_edge_height(edge);
                        if (edge_height === null)
                            throw new Error("unreachable");
                        return <[Edge,HeightRange]>[edge, edge_height];
                    }));
                for (let [edge, edge_height] of below_heights) {
                    if (edge_height.min > height.min + EPSILON) {
                        height.min = edge_height.min;
                        height_changed = true;
                    }
                    this.iffy_history.faces.push(
                        [face, edge, true, height.min, edge_height.min] );
                }
                for (let [edge, edge_height] of above_heights) {
                    if (edge_height.max < height.max - EPSILON) {
                        height.max = edge_height.max;
                        height_changed = true;
                    }
                    this.iffy_history.faces.push(
                        [face, edge, false, height.max, edge_height.max] );
                }
                if (height_changed && (height.max - height.min < EPSILON)) {
                    if (height.max - height.min < -EPSILON) {
                        throw this._inordered_heights_error({face});
                    }
                    height.height = height.max = height.min;
                }
            }
            if (height_changed) {
                this._mark_face_edges_iffy(face);
            }
        }
        this.iffy.faces.clear();
    }
    _inordered_heights_error(additional_info: any | null = null): Error {
        let error: any = new Error("Heights are not ordered correctly");
        error.info = {inordered_heights:
            Object.assign({graph: this}, additional_info) };
        return error;
    }

    check(options?: {[name: string]: boolean, return_errors: true}):
        {errors: any[], info: any[]}
    check(options?: {[name: string]: boolean}): void
    check({
        vertices_without_edges = true,
        edges_with_rogue_vertices = true,
        edges_without_faces = true,
        faces_with_rogue_edges = true,
        above_relation_broken = true,
        height_relation_broken = true,
        return_errors = false,
    }: {[name: string]: boolean} = {}): {errors: any[], info: any[]} | void {
        let {errors, info} = super.check({
            vertices_without_edges,
            edges_with_rogue_vertices,
            edges_without_faces,
            edges_with_one_face: false,
            faces_with_rogue_edges,
            face_orientation: false,
            return_errors: true,
        });
        if (above_relation_broken) {
            let edge_above_face = new BiMap<Edge,Polygon,boolean>();
            for (let edge of this.edges) {
                if (edge.direction === this.direction)
                    continue;
                let {below: below_face, above: above_face} =
                    this.relative_edge_faces(edge);
                if (below_face !== null)
                    edge_above_face.set2(edge, below_face, true);
                if (above_face !== null)
                    edge_above_face.set2(edge, above_face, false);
            }
            let face_above_edge = new BiMap<Polygon,Edge,boolean>();
            for (let face of this.faces) {
                let {below: below_edges, above: above_edges} =
                    this.relative_face_edges(face);
                for (let edge of below_edges)
                    face_above_edge.set2(face, edge, true);
                for (let edge of above_edges)
                    face_above_edge.set2(face, edge, false);

            }
            let broken_incidences = new Array<[Edge,Polygon]>();
            for (let [edge, face, above1] of edge_above_face.entries2()) {
                let above2 = face_above_edge.get2(face, edge);
                if (above2 === undefined || above2 == above1) {
                    broken_incidences.push([edge,face]);
                }
            }
            for (let [face, edge, above2] of face_above_edge.entries2()) {
                let above1 = edge_above_face.get2(edge, face);
                if (above1 === undefined) {
                    broken_incidences.push([edge,face]);
                }
            }
            if (broken_incidences.length > 0) {
                errors.push({above_relation_broken: broken_incidences});
            }
        }
        if (height_relation_broken) {
            let broken_incidences = new Array<[Edge,Polygon]>();
            for (let face of this.faces) {
                let face_height = this.get_face_height(face);
                let {below: below_edges, above: above_edges} =
                    this.relative_face_edges(face);
                for (let edge of below_edges) {
                    let edge_height = this.get_edge_height(edge);
                    if (edge_height === null)
                        throw new Error("unreachable");
                    if (face_height.min < edge_height.min - EPSILON) {
                        broken_incidences.push([edge, face]);
                    }
                }
                for (let edge of above_edges) {
                    let edge_height = this.get_edge_height(edge);
                    if (edge_height === null)
                        throw new Error("unreachable");
                    if (face_height.max > edge_height.max + EPSILON) {
                        broken_incidences.push([edge, face]);
                    }
                }
            }
            if (broken_incidences.length > 0) {
                errors.push({height_relation_broken: broken_incidences});
            }
        }
        if (return_errors)
            return {errors, info};
        if (errors.length > 0) {
            let error: any = new Error("Heighted graph integrity compromised");
            error.info = { HeightedGraph_check:
                Object.assign({}, ...info, ...errors) };
            throw error;
        }
    }
}

