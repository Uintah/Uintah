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

#include <Geom/Geom.h>
#include <Geometry/Point.h>

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

  
#endif
