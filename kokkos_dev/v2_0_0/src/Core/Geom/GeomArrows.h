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
 *  GeomArrows.h: Arrows objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Arrows_h
#define SCI_Geom_Arrows_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {


class SCICORESHARE GeomArrows : public GeomObj {
    double headwidth;
    double headlength;
    Array1<MaterialHandle> shaft_matls;
    Array1<MaterialHandle> back_matls;
    Array1<MaterialHandle> head_matls;
    Array1<Point> positions;
    Array1<Vector> directions;
    Array1<Vector> v1, v2;
    double rad; // radius of arrow shaft if cylinders are drawn
    int drawcylinders; // switch to use lines or cylinders for the arrow
    // The size of the of the head is proportional to the length of the vector.
    // When this flag is set the same size head is used for all the arrows 
    int normalize_headsize;
public:
    GeomArrows(double headwidth, double headlength=0.7, int cyl=0, double r=0,
	       int normhead = 0);
    GeomArrows(const GeomArrows&);
    virtual ~GeomArrows();

    virtual GeomObj* clone();

    void set_matl(const MaterialHandle& shaft_matl,
		  const MaterialHandle& back_matl,
		  const MaterialHandle& head_matl);
    void add(const Point& pos, const Vector& dir);
    void add(const Point& pos, const Vector& dir,
	     const MaterialHandle& shaft, const MaterialHandle& back,
	     const MaterialHandle& head);
    inline int size() { return positions.size(); }

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

// $Log

#endif /* SCI_Geom_Arrows_h */
