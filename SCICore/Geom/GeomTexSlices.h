
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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array3.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::Array3;

class SCICORESHARE GeomTexSlices : public GeomObj {
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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:45  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:08  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:03  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_Grid_h */
