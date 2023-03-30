const POLYGON_SIZE = Object.freeze({
    M: 7, a: 30,
});

document.addEventListener('DOMContentLoaded', function() {
    let canvas_svg = document.getElementById('canvas');
    if (canvas_svg === null || ! (canvas_svg instanceof SVGSVGElement))
        throw new Error();
    let canvas = new Canvas(canvas_svg);
    let viewBox: any;
    let new_canvas = () => {
        canvas = new Canvas();
        document.body.appendChild(canvas.svg);
        canvas.svg.setAttribute('viewBox', viewBox);
    }

    try {
        main(canvas);
    } catch (error: any) {
        viewBox = canvas.svg.getAttribute('viewBox');
        console.log(error);
        console.log(error.info);
        let bad_graph = error?.info?.PlanarGraph?.graph;
        if (bad_graph != null) {
            console.log("drawing graph with a bad construction")
            canvas.draw_graph(bad_graph);
            new_canvas();
        }
        let failed_graph = error?.info?.Graph_check?.graph;
        if (failed_graph != null) {
            console.log("drawing graph with a bad construction")
            let error_data = error.info.Graph_check
            let mask = new Set<Polygon|Edge>();
            for (let edge of <Edge[]>error_data?.edges_without_faces.all || []) {
                canvas.draw_edge(edge, {classes: ["edge", "error_obj"]});
                mask.add(edge);
            }
            for ( let [face] of <Iterable<[Polygon,Edge[]]>>
                error_data?.faces_with_rogue_edges || [] )
            {
                canvas.draw_polygon(face, {classes: ["face", "error_obj"]});
                mask.add(face);
            }
            canvas.draw_graph(failed_graph, mask);
            new_canvas();
        }
        let bad_edge_graph = error?.info?.edge_faces?.graph;
        if (bad_edge_graph != null) {
            console.log("drawing graph with a bad edge")
            let edge = error.info.edge_faces.edge;
            canvas.draw_graph(bad_edge_graph, new Set([edge]));
            canvas.draw_edge( edge,
                {classes: ["edge"],
                    style: {stroke: "cyan", strokeWidth: '0.75px'}} );
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
    const {M, a} = POLYGON_SIZE;
    let {polygon, directions} = Polygon.make_regular(M, a);
    let dir1 = directions[1], dir3 = directions[M-1];
    let vec1 = new DirectedVector(dir1, 0.3*a);
    let vec3 = new DirectedVector(dir3, -0.6*a);
    let vec2 = DirectedVector.make_direction(
        vec1.opposite().add(vec3.opposite()))
    let triangle1 = Polygon.from_vectors( new Point(a, a),
        [vec1, vec2, vec3] );
    let triangle2 = Polygon.from_vectors( new Point(a, 3*a),
        [vec1.opposite(), vec2.opposite(), vec3.opposite()] )
    let canvas_svg = document.getElementById('canvas');
    if (canvas_svg === null || ! (canvas_svg instanceof SVGSVGElement))
        throw new Error();
    canvas.svg.setAttribute( 'viewBox',
        [-(M+1)*a/2, -a/2, (2*M+1)*a/2, (M+1)*a/2].join(" ") );

    let uncut_region = new UncutRegion(polygon, triangle1, triangle2);
    let flow_directions = select_flowing_sector(uncut_region);
    let flows = find_flows(uncut_region, flow_directions);
    let cut_region = construct_cut_region(uncut_region, flows);
}

type FillDrawOptions = {
    classes?: string[],
    style?: {[name: string]: string},
};

class Canvas {
    svg: SVGSVGElement;
    constructor(svg: SVGSVGElement | null = null) {
        if (svg === null)
            svg = makesvg('svg')
        this.svg = svg;
    }
    draw_edge(edge: Edge, options?: FillDrawOptions) {
        makesvg('path', {
            attributes: {
                d: [
                    "M", edge.start.x, edge.start.y,
                    "L", edge.end.x, edge.end.y,
                ].join(" "),
                id: edge.id,
            },
            classes: options?.classes,
            style: options?.style,
            parent: this.svg,
        });
    }
    draw_polygon(polygon: Polygon, options?: FillDrawOptions) {
        let path_items = new Array<string|number>();
        let first_vertex = true;
        for (let vertex of polygon.vertices) {
            path_items.push(first_vertex ? "M" : "L");
            path_items.push(vertex.x, vertex.y);
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
                cx: point.x, cy: point.y,
                r: 0.5,
            },
            classes: ["dot"].concat(options?.classes || []),
            style: options?.style,
            parent: this.svg,
        });
    }
    draw_cut_region(region: CutRegion) {
        let drawn_edges = new Set<Edge>();
        let do_polygon = (polygon: Polygon, options?: FillDrawOptions) => {
            this.draw_polygon(polygon, options);
            for (let edge of polygon)
                drawn_edges.add(edge);
        }
        do_polygon(region.outer_face);
        do_polygon(region.triangle1, {classes: ["start_obj"]});
        do_polygon(region.triangle2, {classes: ["end_obj"]});
        for (let edge of region.graph.edges) {
            if (drawn_edges.has(edge))
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
}

type MakeOptions = {
    classes?: string[] | null,
    attributes?: {[name: string]: any},
    style?: {[name: string]: any},
    text?: string | null,
    children?: HTMLElement[],
    parent?: Node | null,
    namespace?: "http://www.w3.org/2000/svg" | null,
}
type MakeHTMLOptions = MakeOptions & {
    namespace?: null,
}
type MakeSVGOptions = MakeOptions & {
    namespace: "http://www.w3.org/2000/svg",
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
        Object.assign(options, {namespace: "http://www.w3.org/2000/svg"}) );
}

