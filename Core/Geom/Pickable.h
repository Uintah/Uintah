
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

#endif /* SCI_Geom_Pickable_h */
