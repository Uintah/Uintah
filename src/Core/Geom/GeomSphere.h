
/*
 * Sphere.h: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Sphere_h
#define SCI_Geom_Sphere_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/IntVector.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::IntVector;

class SCICORESHARE GeomSphere : public GeomObj {
public:
  Point cen;
  double rad;
  int nu;
  int nv;
  
  void adjust();
  void move(const Point&, double, int nu=20, int nv=10);
  void move(const Point& _cen);
    
  GeomSphere(int nu, int nv, IntVector id);
  GeomSphere(int nu, int nv, int id_int, IntVector id);
  GeomSphere(int nu=20, int nv=10, int id = 0x1234567);
  GeomSphere(const Point&, double, int nu=20, int nv=10, int id = 0x1234567);
  GeomSphere(const Point&, double, int nu, int nv, int id_int, IntVector id);
  GeomSphere(const Point&, double, int nu, int nv, IntVector id);
  GeomSphere(const GeomSphere&);
  virtual ~GeomSphere();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
  virtual bool getId( int& id );
  virtual bool getId( IntVector& id);
  
};

} // End namespace GeomSpace
} // End namespace SCICore

#endif /* SCI_Geom_Sphere_h */
