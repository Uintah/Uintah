/*
 *  GeomTube.h: Tube object
 *
 *  Written by:
 *   Han-Wei Shen
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tube_h 
#define SCI_Geom_Tube_h 1 

#include <Geom/GeomVertexPrim.h>

class SinCosTable;

namespace SCICore {
namespace GeomSpace {

class GeomTube : public GeomVertexPrim {
    int nu;
    Array1<Vector> directions;
    Array1<double> radii;
private:
    void make_circle(int which, Array1<Point>& circle,
		     const SinCosTable& tab); 
public:
    GeomTube(int nu=8); 
    GeomTube(const GeomTube&); 
    virtual ~GeomTube(); 

    virtual GeomObj* clone(); 
    virtual void get_bounds(BBox&); 
    virtual void get_bounds(BSphere&);
  
    void add(GeomVertex*, double, const Vector&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time); 
#endif 
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
// $Log$
// Revision 1.1  1999/07/27 16:56:47  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:10  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:06  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//


#endif /*SCI_Geom_Tube_h */
