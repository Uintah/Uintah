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
 *  Disc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Disc_h
#define SCI_Geom_Disc_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

class SCICORESHARE GeomDisc : public GeomObj {
    Vector v1;
    Vector v2;
    Vector zrotaxis;
    double zrotangle;
public:
    Point cen;
    Vector n;
    double rad;
    int nu;
    int nv;

    void adjust();
    void move(const Point&, const Vector&, double, int nu=20, int nv=2);

    GeomDisc(int nu=20, int nv=2);
    GeomDisc(const Point&, const Vector&, double, int nu=20, int nv=2);
    GeomDisc(const GeomDisc&);
    virtual ~GeomDisc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Disc_h */

