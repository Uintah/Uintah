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
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>

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
    
  GeomSphere(int nu=20, int nv=10);
  GeomSphere(const Point &location, double radius, int nu=20, int nv=10);
  GeomSphere(const GeomSphere &copy);
  virtual ~GeomSphere();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // This is a helper function which determins the nu and nv given an
  // approximate number of polygons desired.
  static void getnunv(const int num_polygons, int &nu, int &nv);
};


class SCICORESHARE GeomSpheres : public GeomObj {
private:
  vector<Point> centers_;
  vector<double> radii_;
  vector<unsigned char> colors_;
  vector<float> indices_;
  int nu_;
  int nv_;
  double global_radius_;
  
public:

  GeomSpheres(double radius = 1.0, int nu=8, int nv=8);
  GeomSpheres(const GeomSpheres &copy);
  virtual ~GeomSpheres();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  void add(const Point &center);
  void add(const Point &center, const MaterialHandle &mat);
  void add(const Point &center, float index);

  // If radius is too small, the sphere is not added and false is returned.
  bool add_radius(const Point &cen, double radius);
  bool add_radius(const Point &cen, double radius, const MaterialHandle &mat);
  bool add_radius(const Point &cen, double radius, float index);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif /* SCI_Geom_Sphere_h */
