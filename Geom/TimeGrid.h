
/*
 *  Grid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_TimeGeom_Grid_h
#define SCI_TimeGeom_Grid_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/Array2.h>

class ColorMap;

class TimeGrid : public GeomObj {
    Array1<float *> tmap;

    float *bmap; // blend map...

    int dimU,dimV;
    int tmap_size; // all the same size!!!

    Array1<int>    tmap_dlist;
    Array1<double> time;

    Point corner;
    Vector u, v, w; // all the same!!!
    int active;
    void adjust();
public:
    TimeGrid(int,int, int, const Point&, const Vector&, const Vector&);
    TimeGrid(const TimeGrid&);
    virtual ~TimeGrid();

    virtual GeomObj* clone();

    // methor for textur mapping...

    ColorMap* map; // color map to be used...

    void set_active(int, double);

    void set(int,int, const MaterialHandle&, const double&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_Grid_h */
