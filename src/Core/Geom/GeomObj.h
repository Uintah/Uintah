
/*
 *  GeomObj.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GeomObj_h
#define SCI_Geom_GeomObj_h 1

#include <Containers/Array1.h>
#include <Containers/Handle.h>
#include <Persistent/Persistent.h>
#include <config.h>

#ifdef KCC
#include <iosfwd.h>  // Forward declarations for KCC C++ I/O routines
#else
class istream;
#endif

namespace SCICore {
  namespace GeomSpace {
    class GeomObj;
  }
  namespace PersistentSpace {
    class Piostream;
    void Pio( Piostream &, GeomSpace::GeomObj *& );
  }
  namespace Containers {
    class clString;
  }
  namespace Geometry {
    class BBox;
    class BSphere;
    class Vector;
    class Point;
    class Ray;
  }
}

namespace SCICore {
namespace GeomSpace {

struct DrawInfoOpenGL;
struct DrawInfoX11;
class  Material;
class  GeomSave;
class  Hit;

using SCICore::PersistentSpace::Persistent;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Containers::Array1;
using SCICore::Geometry::BBox;
using SCICore::Geometry::BSphere;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Ray;
using SCICore::Containers::clString;

class GeomObj : public Persistent {
protected:
    GeomObj* parent;
public:
    GeomObj();
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual GeomObj* clone() = 0;
    void set_parent(GeomObj*);

    virtual void reset_bbox();
    virtual void get_bounds(BBox&) = 0;
    virtual void get_bounds(BSphere&) = 0;

    // For OpenGL
#ifdef SCI_OPENGL
    void pre_draw(DrawInfoOpenGL*, Material*, int lit);
    virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
#endif

    // For X11
    void draw(DrawInfoX11*, Material*);
    virtual void objdraw(DrawInfoX11*, Material*);
    virtual double depth(DrawInfoX11*);
    virtual void get_hit(Vector&, Point&);

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree) = 0;

    // For Raytracing
    virtual void preprocess()=0;
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit)=0;
    virtual Vector normal(const Point& p, const Hit&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    virtual bool saveobj(ostream&, const clString& format, GeomSave*)=0;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:05  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:58  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//

#endif // ifndef SCI_Geom_GeomObj_h


