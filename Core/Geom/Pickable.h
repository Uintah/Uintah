
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

#include <Core/share/share.h>

namespace SCIRun {

class Vector;
class Point;
class ViewWindow;
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

  virtual void geom_pick(GeomPick*, ViewWindow*, int, const BState& bs) = 0;
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

} // End namespace SCIRun

#endif /* SCI_Geom_Pickable_h */
