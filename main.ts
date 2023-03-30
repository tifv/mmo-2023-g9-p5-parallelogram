const POLYGON_SIZE = Object.freeze({
    M: 7, r: 80,
});

const SVGNS = "http://www.w3.org/2000/svg";

document.addEventListener('DOMContentLoaded', function() {
    const {M, r} = POLYGON_SIZE;
    let canvas_svg = document.getElementById('canvas');
    if (canvas_svg === null || ! (canvas_svg instanceof SVGSVGElement))
        throw new Error();
    let canvas = new Canvas(canvas_svg);
    let set_canvas_viewBox = () => {
        canvas.svg.setAttribute( 'viewBox',
        [-1.2*r, -1.2*r, 2.4*r, 2.4*r].join(" ") );
    };
    set_canvas_viewBox()
    let new_canvas = () => {
        canvas = new Canvas();
        document.body.appendChild(canvas.svg);
        set_canvas_viewBox();
    }

    try {
        main(canvas);
    } catch (error: any) {
        console.log(error);
        console.log(error.info);
        let uncut_region = error?.info?.find_flows?.uncut_region;
        if (uncut_region != null) {
            console.log("drawing uncut region")
            canvas.draw_uncut_region(uncut_region);
            new_canvas();
        }
        let failed_graph_info = error?.info?.PlanarGraph;
        if (failed_graph_info?.graph != null) {
            console.log("drawing graph that failed to construct")
            canvas.draw_graph(failed_graph_info.graph);
            new_canvas();
        }
        let bad_graph_info = error?.info?.Graph_check;
        if (bad_graph_info?.graph != null) {
            console.log("drawing graph that failed checks")
            let mask = new Set<Polygon|Edge>();
            for (let edge of <Edge[]>bad_graph_info?.edges_without_faces?.all || []) {
                canvas.draw_edge(edge, {classes: ["edge", "error_obj"]});
                mask.add(edge);
            }
            for ( let [face] of <Iterable<[Polygon,Edge[]]>>
                bad_graph_info?.faces_with_rogue_edges || [] )
            {
                canvas.draw_polygon(face, {classes: ["face", "error_obj"]});
                mask.add(face);
            }
            canvas.draw_graph(bad_graph_info.graph, mask);
            new_canvas();
        }
        let bad_edge_info = error?.info?.PLanarGraph_edge_faces;
        if (bad_edge_info?.graph != null) {
            console.log("drawing graph with a bad edge")
            let edge = bad_edge_info.edge;
            canvas.draw_graph(bad_edge_info.graph, new Set([edge]));
            canvas.draw_edge( edge,
                {classes: ["edge"],
                    style: {stroke: "cyan", strokeWidth: '0.75px'}} );
            new_canvas();
        }
        let bad_heighted_graph =
            error?.info?.inordered_heights?.graph ||
            error?.info?.incut?.heighted_graph ||
            error?.info?.HeightedGraph_check?.graph;
        if (bad_heighted_graph != null) {
            console.log("drawing graph with (maybe) incorrectly ordered heights")
            canvas.draw_heighted_graph(bad_heighted_graph);
            new_canvas();
        }
        let region_history = error?.info?.construct_cut_region?.region_history;
        if (region_history != null) {
            console.log("drawing successful regions")
            for (let {region, sector_start, sector_end}
                of Array.from<any>(region_history).reverse() )
            {
                canvas.draw_cut_region(region);
                canvas.mark_dot(sector_start, {classes: ["start_obj"]});
                canvas.mark_dot(sector_end, {classes: ["end_obj"]});
                new_canvas();
            }
        }
        throw error;
    }
});

function main(canvas: Canvas) {
    const {M, r} = POLYGON_SIZE;
    // const a = r * 2 * Math.sin(Math.PI/(2*M));
    let {polygon, directions, side_length: a} = Polygon.make_regular_even(
        new Point(0, 0), M, r );
    let dir1 = directions[1], dir3 = directions[M-1];
    let vec1 = new DirectedVector(dir1, 0.3*a);
    let vec3 = new DirectedVector(dir3, -0.6*a);
    let vec2 = DirectedVector.make_direction(
        vec1.opposite().add(vec3.opposite()))
    let triangle1 = Polygon.from_vectors( new Point(-0.5*a, -a),
        [vec1, vec2, vec3] );
    let triangle2 = Polygon.from_vectors( new Point(0, +a),
        [vec1.opposite(), vec2.opposite(), vec3.opposite()] )

    let uncut_region = new UncutRegion(polygon, triangle1, triangle2);
    let flow_directions = select_flowing_sector(uncut_region);
    let flows = find_flows(uncut_region, flow_directions);
    let cut_region = construct_cut_region(uncut_region, flows);
    canvas.draw_cut_region(cut_region);
}

type FillDrawOptions = {
    classes?: string[],
    style?: {[name: string]: string},
};

class Canvas {
    svg: SVGSVGElement;
    constructor(svg: SVGSVGElement | null = null) {
        if (svg === null)
            svg = makesvg('svg', {attributes: {
                xmlns: SVGNS,
                width: "500px", height: "500px",
            }});
        this.svg = svg;
    }
    draw_edge(edge: Edge, options?: FillDrawOptions): SVGPathElement {
        return makesvg('path', {
            attributes: {
                d: [
                    "M", edge.start.x, -edge.start.y,
                    "L", edge.end  .x, -edge.end  .y,
                ].join(" "),
            },
            classes: options?.classes,
            style: options?.style,
            parent: this.svg,
        });
    }
    draw_polygon(polygon: Polygon, options?: FillDrawOptions): SVGPathElement {
        let path_items = new Array<string|number>();
        let first_vertex = true;
        for (let vertex of polygon.vertices) {
            path_items.push(first_vertex ? "M" : "L");
            path_items.push(vertex.x, -vertex.y);
            first_vertex = false;
        }
        path_items.push("Z");
        return makesvg('path', {
            attributes: Object.assign({
                d: path_items.join(" "),
            }),
            classes: options?.classes,
            style: options?.style,
            parent: this.svg,
        });
    }
    mark_dot(point: Point, options: FillDrawOptions = {}) {
        makesvg('circle', {
            attributes: {
                cx: point.x, cy: -point.y,
                r: 0.5,
            },
            classes: ["dot"].concat(options?.classes || []),
            style: options?.style,
            parent: this.svg,
        });
    }
    draw_uncut_region(region: UncutRegion) {
        let do_border_face = (polygon: Polygon, options?: FillDrawOptions) => {
            this.draw_polygon(polygon, options);
        }
        do_border_face(region.polygon, {classes:
            ["face", "face__outer", "border", "face__border"] });
        do_border_face(region.triangle1, {classes:
            ["start_obj", "face", "border", "face__border"] });
        do_border_face(region.triangle2, {classes:
            ["end_obj", "face", "border", "face__border"] });
    }
    draw_cut_region(region: CutRegion) {
        let mask = new Set<Edge|Polygon>();
        let do_border_face = (polygon: Polygon, options?: FillDrawOptions) => {
            this.draw_polygon(polygon, options);
            for (let edge of polygon)
                mask.add(edge);
        }
        do_border_face(region.outer_face, {classes:
            ["face", "face__outer", "border", "face__border"] });
        do_border_face(region.triangle1, {classes:
            ["start_obj", "face", "border", "face__border"] });
        do_border_face(region.triangle2, {classes:
            ["end_obj", "face", "border", "face__border"] });
        for (let edge of region.graph.edges) {
            if (mask.has(edge))
                continue;
            this.draw_edge(edge, {classes: ["edge"]});
        }
    }
    draw_graph(
        graph: PlanarGraph,
        mask: Set<Edge|Polygon> | null = null,
    ) {
        for (let face of graph.faces) {
            if (mask !== null && mask.has(face))
                continue;
            let element = this.draw_polygon(face, {classes: ["face"]});
            element.onclick = () => {console.log(face);};
        }
        for (let edge of graph.edges) {
            if (mask !== null && mask.has(edge))
                continue;
            this.draw_edge(edge, {classes: ["edge"]});
        }
    }
    draw_heighted_graph(
        graph: HeightedFaceGraph,
    ) {
        for (let face of graph.faces) {
            let heights = graph.get_face_height(face);
            let height = heights.height !== undefined ? heights.height :
                (heights.min + heights.max) / 2;
            let shift = new DirectedVector(graph.direction, height);
            let drawn_face = Polygon.from_vectors(
                face.vertices[0].shift(shift),
                Array.from(face.oriented_edges()).map(({vector}) => vector) );
            let element = this.draw_polygon(drawn_face, {classes: [
                "face",
                ...(heights.height === undefined ? ["unknown_height"] : []),
            ]});
            element.onclick = () => {console.log({face, heights});};
        }
        for (let edge of graph.edges) {
            let heights = graph.get_edge_height(edge);
            let height: number;
            if (heights !== null) {
                height = heights.height !== undefined ? heights.height :
                    (heights.min + heights.max) / 2;
            } else {
                height = 0;
            }
            let shift = new DirectedVector(graph.direction, height);
            let drawn_edge = edge.shift(shift);
            let element = this.draw_edge(drawn_edge, {classes: [
                "edge",
                ...( (heights === null) ? ["edge__vertical"] :
                    (heights.height === undefined ? ["unknown_height"] : []) ),
            ]});
            element.onclick = () => {console.log({edge, heights});};
        }
    }
}

type MakeOptions = {
    classes?: string[] | null,
    attributes?: {[name: string]: any},
    style?: {[name: string]: any},
    text?: string | null,
    children?: HTMLElement[],
    parent?: Node | null,
    namespace?: typeof SVGNS | null,
}
type MakeHTMLOptions = MakeOptions & {
    namespace?: null,
}
type MakeSVGOptions = MakeOptions & {
    namespace: typeof SVGNS,
}

function makehtml(tag: string, options: MakeHTMLOptions): HTMLElement;
function makehtml(tag: string, options: MakeSVGOptions): SVGElement;

function makehtml(tag: string, {
    classes = null,
    attributes = {},
    style = {},
    text = null,
    children = [],
    parent = null,
    namespace = null,
}: MakeOptions = {}): HTMLElement | SVGElement {
    var element: HTMLElement | SVGElement = ( namespace == null ?
        document.createElement(tag) :
        document.createElementNS(namespace, tag) );
    if (classes) {
        element.classList.add(...classes);
    }
    for (let name in attributes) {
        element.setAttribute(name, attributes[name]);
    }
    for (let name in style) {
        // @ts-ignore
        element.style[name] = style[name];
    }
    if (text != null) {
        element.textContent = text;
    }
    for (let child of children) {
        element.appendChild(child);
    }
    if (parent != null) {
        parent.appendChild(element);
    }
    return element;
}

function makesvg(tag: "svg", options?: MakeOptions): SVGSVGElement
function makesvg(tag: "path", options?: MakeOptions): SVGPathElement
function makesvg(tag: string, options?: MakeOptions): SVGElement

function makesvg(tag: string, options: MakeOptions = {}): SVGElement {
    return makehtml( tag,
        Object.assign(options, {namespace: SVGNS}) );
}

