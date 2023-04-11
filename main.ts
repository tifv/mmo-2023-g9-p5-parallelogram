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
        drag_start: (event) => {
            dragger.start(event);
        }
    });
    let chooser = Choosers.CoprimeChooser.default();
    let reload = () => {
        chooser.reset();
        let flows = uncut_region.find_flows(chooser.offspring());
        let cut_region = construct_cut_region( uncut_region, flows,
            chooser.offspring() );
        drawer.redraw(cut_region);
    };
    let dragger = new Dragger({
        uncut_region,
        drawer,
        reload,
    })

    document.body.addEventListener('mousemove',
        (event) => { dragger.move(event); } );

    document.body.addEventListener('mouseup',
        (event) => { dragger.end(event); } );
    document.body.addEventListener('mouseleave',
        (event) => { dragger.end(event); } );

    document.body.addEventListener('touchmove',
        (event) => { dragger.move(event); } );
    document.body.addEventListener('touchend',
        (event) => { dragger.end(event); } );
    document.body.addEventListener('touchleave',
        (event) => { dragger.end(<MouseEvent|TouchEvent>event); } );
    document.body.addEventListener('touchcancel',
        (event) => { dragger.end(event); } );

    reload();
}

function build_uncut_region(): UncutRegion {
    const {M, r} = POLYGON_SIZE;
    let origin = new Point(0, 0);
    let {polygon, directions, side_length: a} = Polygon.make_regular_even(
        origin, M, r );
    let dir1 = directions[1], dir3 = directions[M-1];
    let vec1 = new DirectedVector(dir1, 0.3*a);
    let vec3 = new DirectedVector(dir3, -0.6*a);
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

    constructor({
        svg: svg,
        drag_start,
    }: {
        svg: SVGSVGElement,
        drag_start: (event: MouseEvent | TouchEvent) => void,
    }) {
        this.svg = svg;
        this.face_group = makesvg("g", {
            parent: this.svg,
            attributes: {
                "stroke": "none",
                "fill": "white",
                "opacity": "0",
        },
        });
        this.edge_group = makesvg("g", {
            parent: this.svg,
            attributes: {
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
                "fill": "rgb(  70%,  85%,  70% )",
            },
        });
        this.triangle2 = makesvg("path", {
            parent: border_group,
            attributes: {
                id: "triangle1",
                "fill": "rgb(  70%,  70%, 100% )",
            },
        });
        this.triangle1.addEventListener('mousedown' , drag_start);
        this.triangle1.addEventListener('touchstart', drag_start);
        this.triangle2.addEventListener('mousedown' , drag_start);
        this.triangle2.addEventListener('touchstart', drag_start);
    };

    svg_coords  = DrawCoords.svg_coords
    math_coords = DrawCoords.math_coords

    redraw(region: CutRegion) {
        this.outer_face.setAttribute( 'd',
            RegionDrawer._face_as_path(region.outer_face) );
        this.triangle1.setAttribute( 'd',
            RegionDrawer._face_as_path(region.triangle1) );
        this.triangle2.setAttribute( 'd',
            RegionDrawer._face_as_path(region.triangle2) );

        let mask = new Set<Edge|Polygon>([
            ...region.outer_face, region.outer_face,
            ...region.triangle1, region.triangle1,
            ...region.triangle2, region.triangle2,
        ]);

        let edge_reuser = new SVGElementReuser<Edge>( this.edge_group,
            (edge, path) => {
                path.setAttribute('d', RegionDrawer._edge_as_path(edge));
            },
            (edge) => makesvg('path', {
                attributes: {
                    d: RegionDrawer._edge_as_path(edge),
                },
            }),
        );
        for (let edge of region.graph.edges) {
            if (mask.has(edge))
                continue;
            edge_reuser.use(edge);
        }
        edge_reuser.clean();

        let face_reuser = new SVGElementReuser<Polygon>( this.face_group,
            (face, path) => {
                path.setAttribute('d', RegionDrawer._face_as_path(face));
            },
            (face) => makesvg('path', {
                attributes: {
                    d: RegionDrawer._face_as_path(face),
                },
            }),
        );
        for (let face of region.graph.faces) {
            if (mask.has(face))
                continue;
            face_reuser.use(face);
        }
        face_reuser.clean();

    }

    static _face_as_path(face: Polygon) {
        let path_items = new Array<string|number>();
        let is_first = true;
        for (let vertex of face.vertices) {
            let coords = DrawCoords.svg_coords(vertex);
            path_items.push(is_first ? "M" : "L");
            path_items.push(coords.x, coords.y);
            is_first = false;
        }
        path_items.push("Z");
        return path_items.join(" ");
    }

    static _edge_as_path(edge: Edge) {
        let
            start = DrawCoords.svg_coords(edge.start),
            end   = DrawCoords.svg_coords(edge.end);
        return [
            "M", start.x, start.y,
            "L", end  .x, end  .y,
        ].join(" ");
    }

}

/** Reuse DOM elements because they are kinda expensive */
class SVGElementReuser<T> {
    protected parent: SVGGElement;
    protected index: number;
    protected reuse: (value: T, element: SVGPathElement) => void;
    protected build: (value: T) => SVGPathElement;

    constructor(
        parent: SVGGElement,
        reuse: (value: T, element: SVGPathElement) => void,
        build: (value: T) => SVGPathElement,
    ) {
        this.parent = parent;
        this.index = 0;
        this.reuse = reuse;
        this.build = build;
    }

    use(value: T): void {
        if (this.index < this.parent.children.length) {
            this.reuse( value,
                <SVGPathElement>this.parent.children[this.index] );
        } else {
            this.parent.appendChild(this.build(value));
        }
        ++this.index;
    }

    clean(): void {
        while (this.parent.children.length > this.index) {
            (<SVGPathElement>this.parent.lastChild).remove();
        }
    }
}

class Dragger {
    uncut_region: UncutRegion;
    drawer: RegionDrawer;

    drag: {
        triangle: 1 | 2,
        offset: Vector,
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
        this.drawer = drawer;
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
    start(event: MouseEvent | TouchEvent) {
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
    start_touch(event: TouchEvent) {
        this.start(event);
    }
    move(event: MouseEvent | TouchEvent): void {
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
    end(event: MouseEvent | TouchEvent) {
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

