const POLYGON_SIZE = Object.freeze({
    M: 7, r: 80,
});

const SVGNS = "http://www.w3.org/2000/svg";

document.addEventListener('DOMContentLoaded', function meta_main() {
    main();
});

function main() {
    let uncut_region = build_uncut_region();
    let [min, max] = DrawCoords.svg_bbox(uncut_region.bbox()),
        size = {x: max.x - min.x, y: max.y - min.y};
    min.x -= 0.1 * size.x; max.x += 0.1 * size.x;
    min.y -= 0.1 * size.y; max.y += 0.1 * size.y;
    size = {x: max.x - min.x, y: max.y - min.y};

    let svg: SVGSVGElement; {
        let element: HTMLElement | null
            = document.getElementById('canvas');
        if (element === null)
            throw new Error("Canvas element not found");
        if (!(element instanceof SVGSVGElement))
            throw new Error("Canvas element is not SVG");
        svg = element;
    }
    svg.setAttribute( 'viewBox',
        [min.x, min.y, size.x, size.y].join(" ") );

    let drawer = new RegionDrawer({
        svg: svg,
    });
    let chooser = Choosers.CoprimeChooser.default();
    let reload = () => {
        chooser.reset();
        let flows = uncut_region.find_flows(chooser.offspring());
        let cut_region = construct_cut_region( uncut_region, flows,
            chooser.offspring() );
        drawer.redraw(cut_region);
    };
    let pointer_watcher = new PointerWatcher({
        uncut_region,
        drawer,
        reload,
    });

    reload();
}

function build_uncut_region(): UncutRegion {
    const {M, r} = POLYGON_SIZE;
    let origin = new Point(0, 0);
    let {polygon, directions, side_length: a} = Polygon.make_regular_even(
        origin, M, r );
    let dir1 = directions[1], dir3 = directions[M-1];
    let vec1 = new DirectedVector(dir1, 0.4*a);
    let vec3 = new DirectedVector(dir3, -0.8*a);
    let vec2 = DirectedVector.make_direction(
        vec1.opposite().add(vec3.opposite()));
    let s = 0.6 + (M - 3) * 0.25;
    let triangle1 = Polygon.from_vectors(
        origin.shift(vec1.scale(-s)).shift(vec3.scale(s)),
        [vec1, vec2, vec3] );
    let triangle2 = Polygon.from_vectors(
        origin.shift(vec1.scale(s)).shift(vec3.scale(-s)),
        [vec1.opposite(), vec2.opposite(), vec3.opposite()] )
    let uncut_region = new UncutRegion(polygon, triangle1, triangle2);
    ( {point1: uncut_region.point1, point2: uncut_region.point2} =
        uncut_region.find_nearest_feasible(
          {close: uncut_region.point1}, {close: uncut_region.point2} )
    );

    return uncut_region;
}

type PointObj = {x: number, y: number};

const DrawCoords = Object.freeze({
    svg_coords: (point: Point): PointObj => {
        return {x: point.x, y: -point.y};
    },
    svg_bbox: ([min, max]: [Point, Point]): [PointObj, PointObj] => {
        let
            svg_min = DrawCoords.svg_coords(min),
            svg_max = DrawCoords.svg_coords(max);
        return [{x: svg_min.x, y: svg_max.y}, {x: svg_max.x, y: svg_min.y}];
    },
    math_coords: (point: PointObj): Point => {
        return new Point(point.x, -point.y);
    },
});

class RegionDrawer {
    svg: SVGSVGElement

    outer_face: SVGPathElement;
    triangle1: SVGPathElement;
    triangle2: SVGPathElement;
    edge_group: SVGGElement;
    face_group: SVGGElement;
    trace_group: SVGGElement;

    graph?: PlanarGraph
    face_by_path: Map<SVGPathElement,Parallelogram> = new Map();
    path_by_face: Map<Parallelogram,SVGPathElement> = new Map();

    constructor({
        svg: svg,
    }: {
        svg: SVGSVGElement,
    }) {
        this.svg = svg;
        this.trace_group = makesvg("g", {
            parent: this.svg,
            attributes: {
                id: "trace_g",
                "stroke": "black",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "fill": "none",
        },
        });
        this.face_group = makesvg("g", {
            parent: this.svg,
            attributes: {
                id: "face_g",
                "stroke": "none",
                "fill": "white",
                "opacity": "0",
        },
        });
        this.edge_group = makesvg("g", {
            parent: this.svg,
            attributes: {
                id: "edge_g",
                "stroke": "black",
                "stroke-width": "0.50px",
                "stroke-linecap": "round",
        },
        });
        let border_group = makesvg("g", {
            parent: this.svg,
            attributes: {
                "stroke": "black",
                "stroke-width": "0.75px",
                "stroke-linejoin": "round",
        },
        });
        this.outer_face = makesvg("path", {
            parent: border_group,
            attributes: {
                id: "outer_face",
                "fill": "none",
            },
        });
        this.triangle1 = makesvg("path", {
            parent: border_group,
            attributes: {
                id: "triangle1",
                "fill": "rgb(  70%,  90%,  70% )",
            },
        });
        this.triangle2 = makesvg("path", {
            parent: border_group,
            attributes: {
                id: "triangle1",
                "fill": "rgb(  70%,  70%, 100% )",
            },
        });
    };

    svg_coords  = DrawCoords.svg_coords
    math_coords = DrawCoords.math_coords

    redraw(region: CutRegion) {
        this.graph = region.graph;

        this.outer_face.setAttribute( 'd',
            RegionDrawer._face_as_d(region.outer_face) );
        this.triangle1.setAttribute( 'd',
            RegionDrawer._face_as_d(region.triangle1) );
        this.triangle2.setAttribute( 'd',
            RegionDrawer._face_as_d(region.triangle2) );

        let mask = new Set<Edge|Polygon>([
            ...region.outer_face, region.outer_face,
            ...region.triangle1, region.triangle1,
            ...region.triangle2, region.triangle2,
        ]);

        new SVGPathProvider<Edge>( this.edge_group,
            (path, edge) => {
                path.setAttribute('d', RegionDrawer._edge_as_d(edge));
            },
        ).use_with_each(filtermap( this.graph.edges, (edge) =>
            mask.has(edge) ? null : edge ));

        this.face_by_path.clear();
        this.path_by_face.clear();
        new SVGPathProvider<Polygon>( this.face_group,
            (path, face) => {
                path.setAttribute('d', RegionDrawer._face_as_d(face));
                let parallelogram = face.as_parallelogram();
                if (parallelogram === null)
                    return;
                this.face_by_path.set(path, parallelogram);
                this.path_by_face.set(parallelogram, path);
            },
        ).use_with_each(filtermap( this.graph.faces, (face) =>
            mask.has(face) ? null : face ));
    }

    redraw_trace(start: {point: Point, face: Parallelogram} | null) {
        if (start === null) {
            SVGPathProvider.clean(this.trace_group);
            this._highlight_faces(new Set());
            return;
        }
        if (this.graph === undefined)
            return;
        let path_provider = new SVGPathProvider( this.trace_group,
            (path, d: string) => { path.setAttribute('d', d); },
        );
        let hightlighted_faces = new Set<Parallelogram>();
        for (let direction of start.face.parallelogram.directions) {
            let points = new Array<Point>();
            for (let forward of [true, false]) {
                let subpoints = new Array<Point>();
                for (let item of this.graph.trace_through_parallelograms(
                    {start, direction, forward},
                )) {
                    if (item instanceof Point) {
                        subpoints.push(item);
                        continue;
                    }
                    hightlighted_faces.add(item);
                }
                if (forward) {
                    points.push(...subpoints);
                } else {
                    points.unshift(...subpoints.reverse());
                }
            }
            path_provider.use_with(RegionDrawer._points_as_d(points));
        }
        path_provider.clean();
        this._highlight_faces(hightlighted_faces);
    }

    _highlight_faces(faces: Set<Parallelogram>) {
        for (let path of this.face_group.children) {
            if (!(path instanceof SVGPathElement))
                continue;
            let face = this.face_by_path.get(path);
            if (face === undefined)
                continue;
            if (faces.has(face)) {
                path.classList.add('highlighted');
            } else {
                path.classList.remove('highlighted');
            }
        }
    }

    static _face_as_d(face: Polygon): string {
        return this._points_as_d(face.vertices, true);
    }

    static _edge_as_d(edge: Edge): string {
        let
            start = DrawCoords.svg_coords(edge.start),
            end   = DrawCoords.svg_coords(edge.end);
        return [
            "M", start.x, start.y,
            "L", end  .x, end  .y,
        ].join(" ");
    }

    static _points_as_d(
        points: Array<Point>,
        cycle: boolean = false,
    ): string {
        let path_items = new Array<string|number>();
        let is_first = true;
        for (let point of points) {
            let coords = DrawCoords.svg_coords(point);
            path_items.push(is_first ? "M" : "L");
            path_items.push(coords.x, coords.y);
            is_first = false;
        }
        if (cycle)
            path_items.push("Z");
        return path_items.join(" ");
    }

}

/** Reuse DOM elements because they are kinda expensive */
class SVGPathProvider<T> {
    protected parent: SVGGElement;
    protected index: number;
    protected _use: (element: SVGPathElement, value: T) => void;

    constructor(
        parent: SVGGElement,
        use: (element: SVGPathElement, value: T) => void,
    ) {
        this.parent = parent;
        this.index = 0;
        this._use = use;
    }

    use_with(value: T): void {
        let element: SVGPathElement;
        while (true) {
            if (this.index >= this.parent.children.length) {
                this.parent.appendChild(element = makesvg('path'));
                ++this.index;
                break;
            }
            let e = this.parent.children[this.index++];
            if (e instanceof SVGPathElement) {
                element = e;
                break;
            }
        }
        this._use(element, value);
    }

    use_with_each(values: Iterable<T>): void {
        for (let value of values)
            this.use_with(value);
        this.clean();
    }

    clean(): void {
        while (this.parent.children.length > this.index) {
            let element = this.parent.lastChild;
            if (element === null)
                break;
            element.remove();
        }
    }

    static clean(parent: SVGGElement): void {
        new SVGPathProvider<void>(parent, () => {}).clean();
    }
}

// fix a lack in ts 4.9.5
interface HTMLElement {
    addEventListener<K extends "touchleave">(
        type: K,
        listener: (this: HTMLElement, ev: TouchEvent) => any,
        options?: boolean | AddEventListenerOptions,
    ): void;
}

class PointerWatcher {
    uncut_region: UncutRegion;
    drawer: RegionDrawer;
    container: HTMLElement;

    drag: {
        triangle: 1 | 2,
        offset: Vector,
    } | null = null;

    trace: {
        point: Point,
        face: Parallelogram,
        cancel: () => void,
        clear: () => void,
    } | null = null;

    reload: () => void;

    constructor( {
        uncut_region,
        drawer,
        reload,
    }: {
        uncut_region: UncutRegion,
        drawer: RegionDrawer,
        reload: () => void,
    }) {
        this.uncut_region = uncut_region;

        let
            drag_start = this.drag_start.bind(this),
            drag_move  = this.drag_move .bind(this),
            drag_end   = this.drag_end  .bind(this),
            face_move  = this.face_move .bind(this);

        this.drawer = drawer;
        this.drawer.triangle1.addEventListener('mousedown' , drag_start);
        this.drawer.triangle1.addEventListener('touchstart', drag_start);
        this.drawer.triangle2.addEventListener('mousedown' , drag_start);
        this.drawer.triangle2.addEventListener('touchstart', drag_start);

        this.drawer.face_group.addEventListener('mousemove', face_move);
        this.drawer.face_group.addEventListener('click'    , face_move);

        this.container = document.body;
        this.container.addEventListener('mousemove'  , drag_move);
        this.container.addEventListener('mouseup'    , drag_end );
        this.container.addEventListener('mouseleave' , drag_end );
        this.container.addEventListener('touchmove'  , drag_move);
        this.container.addEventListener('touchend'   , drag_end );
        this.container.addEventListener('touchleave' , drag_end );
        this.container.addEventListener('touchcancel', drag_end );
        this.reload = reload;
    }

    svg_coords  = DrawCoords.svg_coords
    math_coords = DrawCoords.math_coords

    _get_pointer_coords(event: MouseEvent | TouchEvent): Point {
        let ctm = this.drawer.svg.getScreenCTM();
        if (ctm === null)
            throw new Error("unreachable (hopefully)");
        let pointer = event instanceof MouseEvent ?
            event : event.touches[0];
        return this.math_coords({
            x: (pointer.clientX - ctm.e) / ctm.a,
            y: (pointer.clientY - ctm.f) / ctm.d,
        });
    }
    drag_start(event: MouseEvent | TouchEvent) {
        let target = event.target;
        let triangle: 1|2;
        if (target === this.drawer.triangle1) {
            triangle = 1;
        } else if (target === this.drawer.triangle2) {
            triangle = 2;
        } else {
            return;
        }
        let point = triangle === 1 ?
            this.uncut_region.point1 : this.uncut_region.point2;
        this.drag = { triangle, offset:
            Vector.between(this._get_pointer_coords(event), point) };
    }
    face_move(event: MouseEvent | TouchEvent): void {
        if (this.drag !== null)
            return;
        let target = event.target;
        if (target === null || !(target instanceof SVGPathElement))
            return;
        let path = target;
        let face = this.drawer.face_by_path.get(path);
        if (face === undefined)
            return;
        let point = this._get_pointer_coords(event);
        if (this.trace) {
            this.trace.cancel();
        }
        let trace = this.trace = {
            point, face,
            cancel: () => {
                path.removeEventListener('mouseleave', trace.clear);
                this.trace = null;
            },
            clear: () => {
                if (this.trace !== trace)
                    return;
                this.drawer.redraw_trace(null);
                trace.cancel();
            },
        }
        this.drawer.redraw_trace({point, face});
        path.addEventListener('mouseleave', trace.clear);
    }
    drag_move(event: MouseEvent | TouchEvent): void {
        if (this.drag === null)
            return;
        let point = this._get_pointer_coords(event).shift(
            this.drag.offset );
        if (this.drag.triangle === 1) {
            let {point1} = this.uncut_region.find_nearest_feasible(
                {close: point}, {exact: true} );
            this.uncut_region.point1 = point1;
            this.reload();
        } else {
            let {point2} = this.uncut_region.find_nearest_feasible(
                {exact: true}, {close: point} );
            this.uncut_region.point2 = point2;
            this.reload();
        }
    }
    drag_end(event: MouseEvent | TouchEvent) {
        this.drag = null;
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
        element.style.setProperty(name, style[name]);
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
function makesvg(tag: "g", options?: MakeOptions): SVGGElement
function makesvg(tag: "path", options?: MakeOptions): SVGPathElement
function makesvg(tag: "circle", options?: MakeOptions): SVGCircleElement
function makesvg(tag: string, options?: MakeOptions): SVGElement

function makesvg(tag: string, options: MakeOptions = {}): SVGElement {
    return makehtml( tag,
        Object.assign(options, {namespace: SVGNS}) );
}

