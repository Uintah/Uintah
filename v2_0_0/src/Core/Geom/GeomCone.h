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
 *  Cone.h: Cone object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Cone_h
#define SCI_Geom_Cone_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/Material.h>

namespace SCIRun {

class SCICORESHARE GeomCone : public GeomObj {
protected:
    Vector v1;
    Vector v2;
    double tilt;
    double height;
    Vector zrotaxis;
    double zrotangle;
public:
    Point bottom;
    Point top;
    Vector axis;
    double bot_rad;
    double top_rad;
protected:
    int nu;
    int nv;
public:
    void adjust();
    void move(const Point&, const Point&, double, double, int nu=20, int nv=1);

    GeomCone(int nu=20, int nv=1);
    GeomCone(const Point&, const Point&, double, double, int nu=20, int nv=1);
    GeomCone(const GeomCone&);
    virtual ~GeomCone();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SCICORESHARE GeomCappedCone : public GeomCone {
    int nvdisc1;
    int nvdisc2;
public:
    GeomCappedCone(int nu=20, int nv=1, int nvdisc1=1, int nvdisc2=1);
    GeomCappedCone(const Point&, const Point&, double, double, 
		   int nu=20, int nv=1, int nvdisc1=1, int nvdisc2=1);
    GeomCappedCone(const GeomCappedCone&);
    virtual ~GeomCappedCone();

    virtual GeomObj* clone();
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class SCICORESHARE GeomCones : public GeomObj {
protected:
  double radius_;
  int  nu_;
  vector<Point> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;
  vector<double> radii_;
  
public:
  GeomCones(int nu = 8, double radius = 1.0);
  GeomCones(const GeomCones &copy);
  virtual ~GeomCones();

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

  bool add(const Point &p0, const Point &p1);
  bool add(const Point &p0, const Point &p1, const MaterialHandle &c);
  bool add(const Point &p0, const Point &p1, float index);

  bool add_radius(const Point &p0, const Point &p1, double r);
  bool add_radius(const Point &p0, const Point &p1,
		  const MaterialHandle &c, double r);
  bool add_radius(const Point &p0, const Point &p1, float index, double r);
  void set_radius(double val) { radius_ = val; reset_bbox(); }
  void set_nu(int nu);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


} // End namespace SCIRun


#endif /* SCI_Geom_Cone_h */

