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
 * GeomTorus.h: Torus objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Torus_h
#define SCI_Geom_Torus_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE GeomTorus : public GeomObj {
public:
    Point cen;
    Vector axis;
    double rad1;
    double rad2;
    int nu;
    int nv;

    Vector zrotaxis;
    double zrotangle;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);

    GeomTorus(int nu=50, int nv=8);
    GeomTorus(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);
    GeomTorus(const GeomTorus&);
    virtual ~GeomTorus();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SCICORESHARE GeomTorusArc : public GeomTorus {
public:
    Vector zero;
    double start_angle;
    double arc_angle;
    Vector yaxis;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      const Vector& zero, double start_angle, double arc_angle,
	      int nu=50, int nv=8);
    GeomTorusArc(int nu=50, int nv=8);
    GeomTorusArc(const Point&, const Vector&, double, double, 
		 const Vector& zero, double start_angle, double arc_angle,
		 int nu=50, int nv=8);
    GeomTorusArc(const GeomTorusArc&);
    virtual ~GeomTorusArc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Torus_h */
