
/*
 * Pt.h: Pts objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Feb 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Point_h
#define SCI_Geom_Point_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomPts : public GeomObj {
public:
    Array1<float> pts;
    inline void add(const Point& p) {
	int s=pts.size();
	pts.grow(3);
 	pts[s]=p.x();
	pts[s+1]=p.y();
	pts[s+2]=p.z();
    }
    int have_normal;
    Vector n;
    GeomPts(const GeomPts&);
    GeomPts(int size);
    GeomPts(int size, const Vector &);
    virtual ~GeomPts();
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
    virtual Vector normal(const Point& p, const Hit&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_Point_h */
