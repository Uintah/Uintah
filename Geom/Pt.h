class GeomPt : public GeomObj {
public:
    Point p1;

    GeomPt(const Point&);
    GeomPt(const GeomPt&);
    virtual ~GeomPt();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    virtual void make_prims(Array1<GeomObj*>& free,
		    Array1<GeomObj*>& dontfree);
    virtual double depth(DrawInfo*);
    virtual void objdraw_X11(DrawInfo*);
};

