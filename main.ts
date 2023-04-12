namespace Main {

document.addEventListener('DOMContentLoaded', async function meta_main() {
    await main();
});

async function main(): Promise<void> {
    const M = 7, r = 80;
    let uncut_region = build_uncut_region({M, r});
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

type TriangleIndex = 1 | 2;
const TRIANGLE_INDICES = <TriangleIndex[]>[1, 2];

function build_uncut_region({M, r}: {M: number, r: number}): UncutRegion {
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

type PointObj = {
    x: number,
    y: number,
};

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
    triangles: Record<TriangleIndex,{
        group: SVGGElement,
        face: SVGPathElement,
        hint: SVGElement,
    }>;
    edge_group: SVGGElement;
    face_group: SVGGElement;
    trace_group: SVGGElement;

    graph?: PlanarGraph;
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
                id: "border_g",
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
        let clipper = (r: number) => [
            "M", r, r, "L", -r, r,
            "L", -r, -r, "L", r, -r, "Z",
        ].join(" ");
        makesvg("defs", {
            parent: this.svg,
            children: [
            makesvg("clipPath", {
                parent: border_group,
                attributes: {
                    id: "hint_clip",
                    "clip-rule": "evenodd",
                },
                children: [
                makesvg("path", {attributes: {
                    "d" : clipper(1) + " " + clipper(0.65),
                }}),
                ],
            })
            ],
        });
        let make_triangle_group = (index: TriangleIndex) => {
            let
                group: SVGGElement,
                face: SVGPathElement,
                hint: SVGElement;
            group = makesvg("g", {
                parent: border_group,
                attributes: {
                    id: "triangle" + index,
                    "fill": index === 1 ?
                        "rgb(  70%,  90%,  70% )" : "rgb(  70%,  70%, 100% )",
                },
                classes: ["triangle_group"],
                children: [
                    face = makesvg("path"),
                    hint = makesvg('g', {
                        attributes: {
                            "visibility": "hidden",
                        },
                        classes: ["hint"],
                        children: [
                        makesvg("path", {
                            attributes: {
                                "d": "M 1 0 L 0 1 L -1 0 L 0 -1 Z",
                                "clip-path": "url(#hint_clip)",
                            },
                            classes: ["hint--inner"],
                        }),
                        ],
                    }),
                ],
            });
            return { group, face, hint };
        }
        this.triangles = {
            1: make_triangle_group(1),
            2: make_triangle_group(2),
        };
    };

    svg_coords  = DrawCoords.svg_coords
    math_coords = DrawCoords.math_coords

    redraw(region: CutRegion) {
        this.graph = region.graph;

        this.outer_face.setAttribute( 'd',
            RegionDrawer._face_as_d(region.outer_face) );
        for (let index of TRIANGLE_INDICES) {
            let polygon = region.triangles[index];
            this.triangles[index].face.setAttribute( 'd',
                RegionDrawer._face_as_d(polygon),
            );
            let hint = this.triangles[index].hint;
            let hint_ok = false;
            do_hint: {
                let vertices = polygon.get_sides().map(({start}) => start);
                if (vertices.length !== 3)
                    break do_hint;
                let [a, b, c] = vertices
                let {incenter, inradius} = Point.incenter(a, b, c);
                let coords = DrawCoords.svg_coords(incenter);
                hint.setAttribute('transform', [
                    "translate(" + coords.x + " " + coords.y + ")",
                    "scale(" + (inradius / 1.5) + ")",
                ].join(" "));
                hint.classList.add("hint__visible");
                hint_ok = true;
            }
            if (!hint_ok)
                hint.classList.remove("hint__visible");
        }

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

    private _requested_region: {region: CutRegion} | null = null;

    request_redraw(region: CutRegion): void {
        this._requested_region = {region};
        window.requestAnimationFrame(() => {
            if (this._requested_region === null)
                return;
            this.redraw(this._requested_region.region);
            this._requested_region = null;
        })
    }

    redraw_trace(
        start: {point: Point, face: Parallelogram} | null,
    ): void {
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

    private _requested_trace: {start: Parameters<
        (typeof RegionDrawer.prototype.redraw_trace)
    >[0]} | null = null;

    request_redraw_trace(
        start: {point: Point, face: Parallelogram} | null,
    ): void {
        this._requested_trace = {start};
        window.requestAnimationFrame(() => {
            if (this._requested_trace === null)
                return;
            this.redraw_trace(this._requested_trace.start);
            this._requested_trace = null;
        })
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

class PointerWatcher {
    uncut_region: UncutRegion;
    drawer: RegionDrawer;
    container: HTMLElement;

    drag: {
        triangle: TriangleIndex,
        offset: Vector,
        group: SVGGElement,
    } | null = null;

    trace: {
        point: Point,
        face: Parallelogram,
        cancel: () => void,
        clear: () => void,
    } | null = null;

    reload: () => void;

    listeners: {
        drag_start: (event: MouseEvent | TouchEvent) => void,
        drag_move : (event: MouseEvent | TouchEvent) => void,
        drag_end  : (event: MouseEvent | TouchEvent) => void,
        face_move : (event: MouseEvent | TouchEvent) => void,
    };

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

        let listeners = this.listeners = {
            drag_start: this.drag_start.bind(this),
            drag_move : this.drag_move .bind(this),
            drag_end  : this.drag_end  .bind(this),
            face_move : this.face_move .bind(this),
        }

        this.drawer = drawer;

        for (let index of TRIANGLE_INDICES) {
            let triangle = this.drawer.triangles[index].group;
            triangle.addEventListener('mousedown'  , listeners.drag_start);
            triangle.addEventListener( 'touchstart', listeners.drag_start,
                {passive: false} );
        }

        let faces = this.drawer.face_group;
        faces.addEventListener('mousemove'  , listeners.face_move);
        faces.addEventListener('click'      , listeners.face_move);
        faces.addEventListener( 'touchstart', listeners.face_move,
            {passive: false} );
        faces.addEventListener( 'touchmove' ,
            (event) => { event.preventDefault(); },
            {passive: false} );

        this.container = document.body;
        this.reload = reload;
    }

    svg_coords  = DrawCoords.svg_coords
    math_coords = DrawCoords.math_coords

    _get_pointer_coords(event: (
        MouseEvent | TouchEvent |
        {clientX: number, clientY: number}
    )): Point {
        let ctm = this.drawer.svg.getScreenCTM();
        if (ctm === null)
            throw new Error("unreachable (hopefully)");
        let pointer = event instanceof TouchEvent ?
            event.touches[0] : event;
        return this.math_coords({
            x: (pointer.clientX - ctm.e) / ctm.a,
            y: (pointer.clientY - ctm.f) / ctm.d,
        });
    }
    drag_start(event: MouseEvent | TouchEvent) {
        if (event instanceof MouseEvent && event.buttons !== 1)
            return;
        let target = event.currentTarget;
        let triangle: (
            (typeof this.drawer.triangles)[TriangleIndex] &
            {index: TriangleIndex}
        ) | null = null
        for (let index of TRIANGLE_INDICES) {
            let t = this.drawer.triangles[index];
            if (target === t.group) {
                triangle = Object.assign({}, t, {index});
            }
        }
        if (triangle === null)
            return;
        event.preventDefault();
        if (this.trace) {
            this.trace.clear();
        }
        let point = this.uncut_region.points[triangle.index];
        this.drag_add_events();
        this.drag = {
            triangle: triangle.index,
            offset: Vector.between(this._get_pointer_coords(event), point),
            group: triangle.group,
        };
        triangle.group.classList.add("triangle_group__dragged");
    }
    drag_add_events() {
        let {container, listeners} = this;
        container.addEventListener('mousemove'   , listeners.drag_move);
        container.addEventListener('mouseup'     , listeners.drag_end );
        container.addEventListener('mouseleave'  , listeners.drag_end );
        container.addEventListener( 'touchmove'  , listeners.drag_move,
            {passive: false} );
        container.addEventListener( 'touchend'   , listeners.drag_end ,
            {passive: false} );
        container.addEventListener( 'touchleave' , listeners.drag_end ,
            {passive: false} );
        container.addEventListener( 'touchcancel', listeners.drag_end ,
            {passive: false} );
    }
    drag_remove_events() {
        let {container, listeners} = this;
        container.removeEventListener('mousemove'  , listeners.drag_move);
        container.removeEventListener('mouseup'    , listeners.drag_end );
        container.removeEventListener('mouseleave' , listeners.drag_end );
        container.removeEventListener('touchmove'  , listeners.drag_move);
        container.removeEventListener('touchend'   , listeners.drag_end );
        container.removeEventListener('touchleave' , listeners.drag_end );
        container.removeEventListener('touchcancel', listeners.drag_end );
    }
    drag_move(event: MouseEvent | TouchEvent): void {
        if (this.drag === null)
            return;
        event.preventDefault();
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
        if (this.drag === null)
            return;
        this.drag.group.classList.remove("triangle_group__dragged");
        this.drag = null;
        this.drag_remove_events();
    }
    face_move(event: MouseEvent | TouchEvent): void {
        if (this.drag !== null)
            return;
        let target = event.target;
        if (target === null || !(target instanceof SVGPathElement))
            return;
        event.preventDefault();
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
                listener_removers.forEach(x => x());
                this.trace = null;
            },
            clear: () => {
                if (this.trace !== trace)
                    return;
                this.drawer.redraw_trace(null);
                trace.cancel();
            },
        }
        let listener_removers = new Array<() => void>();
        this.drawer.redraw_trace({point, face});
        if (event instanceof MouseEvent) {
            path.addEventListener('mouseleave', trace.clear);
            listener_removers.push( () =>
                path.removeEventListener('mouseleave', trace.clear) );
        } else if (event instanceof TouchEvent) {
            // no-op for now. Maybe here will appear a better strategy.
        }
    }
}

const SVGNS = "http://www.w3.org/2000/svg";

type MakeOptions = {
    classes?: string[] | null,
    attributes?: {[name: string]: any},
    style?: {[name: string]: any},
    text?: string | null,
    children?: (HTMLElement|SVGElement)[],
    parent?: Node | null,
    namespace?: typeof SVGNS | null,
}
type MakeHTMLOptions = MakeOptions & {
    children?: HTMLElement[],
    namespace?: null,
}
type MakeSVGOptions = MakeOptions & {
    children?: SVGElement[],
}
type _MakeSVGOptions = MakeSVGOptions & {
    children?: SVGElement[],
    namespace: typeof SVGNS,
}

function makehtml(tag: string, options: MakeHTMLOptions): HTMLElement;
function makehtml(tag: string, options: _MakeSVGOptions): SVGElement;

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

function makesvg(tag: "svg",    options?: MakeSVGOptions): SVGSVGElement
function makesvg(tag: "g",      options?: MakeSVGOptions): SVGGElement
function makesvg(tag: "path",   options?: MakeSVGOptions): SVGPathElement
function makesvg(tag: "circle", options?: MakeSVGOptions): SVGCircleElement
function makesvg(tag: "text",   options?: MakeSVGOptions): SVGTextElement
function makesvg(tag: string,   options?: MakeSVGOptions): SVGElement

function makesvg(tag: string, options: MakeSVGOptions = {}): SVGElement {
    return makehtml( tag,
        Object.assign(options, {namespace: SVGNS}) );
}

}

// fix a lack in ts definitions
interface HTMLElement {
    addEventListener<K extends "touchleave">(
        type: K,
        listener: (this: HTMLElement, ev: TouchEvent) => any,
        options?: boolean | AddEventListenerOptions,
    ): void;
    removeEventListener<K extends "touchleave">(
        type: K,
        listener: (this: HTMLElement, ev: TouchEvent) => any,
        options?: boolean | EventListenerOptions,
    ): void;
}

