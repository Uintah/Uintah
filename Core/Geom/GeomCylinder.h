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


class SCICORESHARE GeomCylinder : public GeomObj {
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


class SCICORESHARE GeomCylinders : public GeomObj {
protected:
  double radius_;
  int  nu_;
  vector<Point> points_;
  vector<MaterialHandle> colors_;
  vector<float> indices_;

public:
  GeomCylinders(int nu = 8, double radius = 1.0);
  GeomCylinders(const GeomCylinders &copy);
  virtual ~GeomCylinders();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

  void add(const Point &p0, const Point &p1);
  void add(const Point &p0, MaterialHandle c0,
	   const Point &p1, MaterialHandle c1);
  void add(const Point &p0, float index0,
	   const Point &p1, float index1);
  void set_radius(double val) { radius_ = val; reset_bbox(); }
  void set_nu_nv(int nu, int nv);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class SCICORESHARE GeomCappedCylinder : public GeomCylinder {
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


class SCICORESHARE GeomCappedCylinders : public GeomCylinders {
public:
  GeomCappedCylinders(int nu = 8, double radius = 1.0);
  GeomCappedCylinders(const GeomCappedCylinders &copy);
  virtual ~GeomCappedCylinders();

  virtual GeomObj* clone();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


    
} // End namespace SCIRun


#endif /* SCI_Geom_Cylinder_h */
