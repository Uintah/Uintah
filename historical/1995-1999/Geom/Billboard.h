
/*
 *  Billboard.h: Billboard object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Geom_Billboard_h
#define SCI_Geom_Billboard_h 1

#include <Geom/Geom.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

class GeomBillboard: public GeomObj {
  GeomObj* child;
  Point at;
  
  BBox bbox;
  BSphere bsphere;
public:
  GeomBillboard(GeomObj*, const Point &);
  
  virtual ~GeomBillboard();
  
  virtual GeomObj* clone();
  //    virtual void reset_bbox();
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BSphere&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>& dontfree);
  virtual void preprocess();
  virtual void intersect(const Ray& ray, Material*,
			 Hit& hit);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;	
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif
