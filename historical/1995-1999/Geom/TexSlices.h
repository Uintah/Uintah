
/*
 *  GeomTexSlices.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_GeomTexSlices_h
#define SCI_Geom_GeomTexSlices_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/Array3.h>

class GeomTexSlices : public GeomObj {
    Point min, max;
    int nx, ny, nz;
    unsigned int texids[3];
    int have_drawn;
public:
    double accum;
    double bright;
    Array3<char> Xmajor;
    Array3<char> Ymajor;
    Array3<char> Zmajor;

    GeomTexSlices(int, int, int, const Point&, const Point&);
    GeomTexSlices(const GeomTexSlices&);
    virtual ~GeomTexSlices();

    virtual GeomObj* clone();

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
