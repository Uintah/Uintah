/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


class GeomSphere : public GeomObj {
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


class GeomSuperquadric : public GeomObj {
  int axis_;
  double A_, B_;
  int nu_, nv_;

  vector<float> points_;
  vector<float> normals_;
  vector<unsigned short> tindices_;
  vector<unsigned short> qindices_;

  void compute_geometry();

  GeomSuperquadric();

public:
  GeomSuperquadric(int axis, double A, double B, int nu, int nv);
  GeomSuperquadric(const GeomSuperquadric &copy);
  virtual ~GeomSuperquadric();
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  static Persistent *maker();
};


class GeomSpheres : public GeomObj {
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
