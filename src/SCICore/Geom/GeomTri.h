
/*
 *  Tri.h: Triangles...
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tri_h
#define SCI_Geom_Tri_h 1

#include <Geom/GeomVertexPrim.h>

namespace SCICore {
namespace GeomSpace {

class GeomTri : public GeomVertexPrim {
    int x_cross(double p1[2], double p2[2], double p[2]);
    Vector n;
public:
    GeomTri(const Point&, const Point&, const Point&);
    GeomTri(const Point&, const Point&, const Point&,
	    const MaterialHandle&,
	    const MaterialHandle&,
	    const MaterialHandle&);
    GeomTri(const GeomTri&);
    virtual ~GeomTri();

#ifdef BSPHERE
    virtual void get_bounds(BSphere& bs);
#endif

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void objdraw(DrawInfoX11*, Material*);
    virtual double depth(DrawInfoX11*);
    virtual void get_hit(Vector&, Point&);
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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:46  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:09  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:05  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:18  dav
// Import sources
//
//

#endif /* SCI_Geom_Tri_h */
