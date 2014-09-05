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
 *  Cylinder.h: Cylinder Object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Cylinder_h
#define SCI_Geom_Cylinder_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {


class GeomCylinder : public GeomObj {
protected:
    Vector v1;
    Vector v2;

    double height;
    Vector zrotaxis;
    double zrotangle;
public:
    Point bottom;
    Point top;
    Vector axis;
    double rad;
    int nu;
    int nv;
    void adjust();
    void move(const Point&, const Point&, double, int nu=20, int nv=1);

    GeomCylinder(int nu=20, int nv=1);
    GeomCylinder(const Point&, const Point&, double, int nu=20, int nv=1);
    GeomCylinder(const GeomCylinder&);
    virtual ~GeomCylinder();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomCylinders : public GeomObj {
protected:
  double radius_;
  int  nu_;
  vector<Point> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;

public:
  GeomCylinders(int nu = 8, double radius = 1.0);
  GeomCylinders(const GeomCylinders &copy);
  virtual ~GeomCylinders();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

  bool add(const Point &p0, const Point &p1);
  bool add(const Point &p0, const MaterialHandle &c0,
	   const Point &p1, const MaterialHandle &c1);
  bool add(const Point &p0, float index0,
	   const Point &p1, float index1);
  void set_radius(double val) { radius_ = val; reset_bbox(); }
  void set_nu_nv(int nu, int nv);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomCappedCylinder : public GeomCylinder {
    int nvdisc;
public:
    GeomCappedCylinder(int nu=20, int nv=1, int nvdisc=1);
    GeomCappedCylinder(const Point&, const Point&, double, int nu=20, int nv=1, int nvdisc=1);
    GeomCappedCylinder(const GeomCappedCylinder&);
    virtual ~GeomCappedCylinder();

    virtual GeomObj* clone();
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomCappedCylinders : public GeomCylinders {
  vector<double> radii_;
public:
  GeomCappedCylinders(int nu = 8, double radius = 1.0);
  GeomCappedCylinders(const GeomCappedCylinders &copy);
  virtual ~GeomCappedCylinders();

  virtual GeomObj* clone();

  void add_radius(const Point &p0, const Point &p1, double r);
  void add_radius(const Point &p0, const MaterialHandle &c0,
		  const Point &p1, const MaterialHandle &c1, double r);
  void add_radius(const Point &p0, float index0,
		  const Point &p1, float index1, double r);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


    
} // End namespace SCIRun


#endif /* SCI_Geom_Cylinder_h */
