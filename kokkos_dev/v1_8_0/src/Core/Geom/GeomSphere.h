/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>

namespace SCIRun {


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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
  virtual bool getId( int& id );
  virtual bool getId( IntVector& id);

  // This is a helper function which determins the nu and nv given an
  // approximate number of polygons desired.
  static void getnunv(const int num_polygons, int &nu, int &nv);
};

} // End namespace SCIRun

#endif /* SCI_Geom_Sphere_h */
