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

#include <Geom/GeomObj.h>
#include <Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class GeomSticky : public GeomObj {
  GeomObj *child;
  
public:
  GeomSticky( GeomObj *c );
  GeomSticky(const GeomSticky&);
  virtual ~GeomSticky();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>& dontfree);
  virtual void preprocess();
  virtual void intersect(const Ray& ray, Material*, Hit& hit);
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:51  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:13  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:12  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif
