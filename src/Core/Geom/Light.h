
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

#include <config.h>
#include <Persistent/Persistent.h>
#include <Containers/String.h>

namespace SCICore {

namespace GeomSpace {
  class Light;
}

namespace PersistentSpace {
  class Piostream;
  void Pio( Piostream &, GeomSpace::Light *& );
}

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


class Light : public Persistent {
protected:
    Light(const clString& name);
public:
    clString name;
    virtual ~Light();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&)=0;
    virtual GeomObj* geom()=0;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
    virtual void lintens(const OcclusionData& od, const Point& hit_position,
			 Color& light, Vector& light_dir)=0;
    virtual void io(Piostream&);
    friend void PersistentSpace::Pio(Piostream&, Light*&);
    static PersistentTypeID type_id;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
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

