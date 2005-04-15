
/*
 *  Pickable.h: ???
 *
 *  Written by:
 *   Dav de St. Germain...
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1999
 *
 *  Copyright (C) 1999 University of Utah
 */

#ifndef SCI_Geom_Pickable_h
#define SCI_Geom_Pickable_h 1

#include <SCICore/share/share.h>

namespace PSECommon {
  namespace Modules {
    class Roe;
  }
}

namespace SCICore {

namespace Geometry {
  class Vector;
  class Point;
}

namespace GeomSpace {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

using PSECommon::Modules::Roe;

class GeomPick;
class GeomObj;

struct BState {
   unsigned int control:1;
   unsigned int alt:1;
   unsigned int shift:1;
   unsigned int btn:2;
};

class SCICORESHARE Pickable {

public:
  virtual ~Pickable();

  virtual void geom_pick(GeomPick*, Roe*, int, const BState& bs) = 0;
  //virtual void geom_pick(GeomPick*, void*, int) = 0;
  virtual void geom_pick(GeomPick*, void*, GeomObj*) = 0;
  virtual void geom_pick(GeomPick*, void*) = 0;
  
  virtual void geom_release(GeomPick*, int, const BState& bs) = 0;
  //  virtual void geom_release(GeomPick*, void*, int) = 0;
  virtual void geom_release(GeomPick*, void*, GeomObj*) = 0;
  virtual void geom_release(GeomPick*, void*) = 0;

  virtual void geom_moved(GeomPick*, int, double, const Vector&, void*) = 0;
  //virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, int) = 0;
  virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, GeomObj*) = 0;
  virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&) = 0;
  virtual void geom_moved(GeomPick*, int, double, const Vector&, const BState&, int) = 0;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2.2.2  2000/10/26 17:18:38  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/08/11 15:44:41  bigler
// Changed geom_* functions that took an int index to take a GeomObj* picked_obj.
//
// Revision 1.2  1999/08/17 06:39:21  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:50  mcq
// Initial commit
//
// Revision 1.4  1999/05/13 18:14:04  dav
// updated Pickable to use pure virtual functions
//
// Revision 1.3  1999/05/06 19:56:12  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:10  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Pickable_h */
