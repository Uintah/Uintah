
/*
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <SCICore/share/share.h>

#ifndef _WIN32
#include <config.h>
#endif
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>

namespace SCICore {

namespace Geometry {
  class Point;
  class Vector;
}

namespace GeomSpace {

using SCICore::PersistentSpace::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::PersistentSpace::Piostream;
using SCICore::Containers::clString;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

class Color;
class GeomObj;
class OcclusionData;
class View;
struct DrawInfoOpenGL;


class SCICORESHARE Light : public Persistent {
protected:
    Light(const clString& name);
public:
    clString name;
    virtual ~Light();
    virtual void io(Piostream&);

    friend SCICORESHARE void Pio( Piostream&, Light*& );

    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:31  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:19  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:11  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:09  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Light_h */

