
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



#endif // ifndef SCI_Geom_GeomObj_h
