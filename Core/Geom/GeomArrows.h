
/*
 *  GeomArrows.h: Arrows objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Arrows_h
#define SCI_Geom_Arrows_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::PersistentSpace::Piostream;

class SCICORESHARE GeomArrows : public GeomObj {
    double headwidth;
    double headlength;
    double rad;
    Array1<MaterialHandle> shaft_matls;
    Array1<MaterialHandle> back_matls;
    Array1<MaterialHandle> head_matls;
    Array1<Point> positions;
    Array1<Vector> directions;
    Array1<Vector> v1, v2;
    int drawcylinders;
public:
    GeomArrows(double headwidth, double headlength=0.7, int cyl=0, double r=0);
    GeomArrows(const GeomArrows&);
    virtual ~GeomArrows();

    virtual GeomObj* clone();

    void set_matl(const MaterialHandle& shaft_matl,
		  const MaterialHandle& back_matl,
		  const MaterialHandle& head_matl);
    void add(const Point& pos, const Vector& dir);
    void add(const Point& pos, const Vector& dir,
	     const MaterialHandle& shaft, const MaterialHandle& back,
	     const MaterialHandle& head);
    inline int size() { return positions.size(); }

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log
//

#endif /* SCI_Geom_Arrows_h */
