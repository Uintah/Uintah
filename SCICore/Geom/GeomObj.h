
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

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Handle.h>
#include <SCICore/Persistent/Persistent.h>
#include <config.h>

#ifdef KCC
#include <iosfwd.h>  // Forward declarations for KCC C++ I/O routines
#else
class istream;
#endif

namespace SCICore {

namespace Containers {
  class clString;
}
namespace Geometry {
  class BBox;
  class Vector;
  class Point;
}

namespace GeomSpace {

struct DrawInfoOpenGL;
class  Material;
class  GeomSave;
class  Hit;

using SCICore::PersistentSpace::Persistent;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Containers::Array1;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Containers::clString;

class SCICORESHARE GeomObj : public Persistent {
protected:
public:
    GeomObj();
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual GeomObj* clone() = 0;

    virtual void reset_bbox();
    virtual void get_bounds(BBox&) = 0;

    // For OpenGL
#ifdef SCI_OPENGL
    void pre_draw(DrawInfoOpenGL*, Material*, int lit);
    virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
#endif
    static PersistentTypeID type_id;

    virtual void io(Piostream&);    
    virtual bool saveobj(ostream&, const clString& format, GeomSave*)=0;
};

void Pio(Piostream&, GeomObj*&);

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:22  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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


