
/*
 * Box.h:  Box object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Geom_Box_h
#define SCI_Geom_Box_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>

class GeomBox : public GeomObj {
  Point min, max;
  int opacity[6];
public:

  GeomBox( const Point& p, const Point& q, int op );
  GeomBox(const GeomBox&);
  virtual ~GeomBox();

  int opaque(int i) { return opacity[i]; }
  void opaque( int i, int op ) { opacity[i] = op; }
  virtual GeomObj* clone();
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

#endif /* SCI_Geom_Box_h */
