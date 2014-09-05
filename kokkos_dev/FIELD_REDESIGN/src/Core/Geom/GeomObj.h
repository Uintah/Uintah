
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
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Persistent/Persistent.h>
#include <sci_config.h>

#include <iosfwd>

namespace SCICore {

namespace Containers {
  class clString;
}
namespace Geometry {
  class BBox;
  class Vector;
  class Point;
  class IntVector;
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
using SCICore::Geometry::IntVector;
using SCICore::Containers::clString;

class SCICORESHARE GeomObj : public Persistent {
public:
    GeomObj(int id = 0x1234567);
    GeomObj(IntVector id);
    GeomObj(int id_int, IntVector id);
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual GeomObj* clone() = 0;

    virtual void reset_bbox();
    virtual void get_bounds(BBox&) = 0;

    // For OpenGL
#ifdef SCI_OPENGL
    int pre_draw(DrawInfoOpenGL*, Material*, int lit);
    virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
    int post_draw(DrawInfoOpenGL*);
#endif
    virtual void get_triangles( Array1<float> &v );
    static PersistentTypeID type_id;

    virtual void io(Piostream&);    
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*)=0;
  // we want to return false if value is the default value
    virtual bool getId( int& ) { return false; }
    virtual bool getId( IntVector& ){ return false; }
protected:

  int id;
  IntVector _id;
};

void Pio(Piostream&, GeomObj*&);

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.8.2.2  2000/10/26 17:18:36  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.11  2000/09/11 22:14:46  bigler
// Added constructors that take an int and IntVector to allow unique
// identification in 4 dimensions.
//
// Revision 1.10  2000/08/09 18:21:14  kuzimmer
// Added IntVector indexing to GeomObj & GeomSphere
//
// Revision 1.9  2000/06/06 16:01:45  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.8  2000/01/03 20:12:36  kuzimmer
//  Forgot to check in these files for picking spheres
//
// Revision 1.7  1999/10/16 20:51:00  jmk
// forgive me if I break something -- this fixes picking and sets up sci
// bench - go to /home/sci/u2/VR/PSE for the latest sci bench technology
// gota getup to get down.
//
// Revision 1.6  1999/10/07 02:07:42  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:26:50  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/19 05:30:55  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
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
