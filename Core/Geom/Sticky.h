/*
 *  Sticky.h - Sticky frame for GeomObj's
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   Jone 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_STICKY_H
#define SCI_STICKY_H 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE GeomSticky : public GeomObj {
  GeomObj *child;
  
public:
  GeomSticky( GeomObj *c );
  GeomSticky(const GeomSticky&);
  virtual ~GeomSticky();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif
