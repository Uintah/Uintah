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
 *  Polyline.h: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Polyline_h
#define SCI_Geom_Polyline_h 1

#include <Core/Geom/GeomVertexPrim.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomPolyline : public GeomVertexPrim {
public:
    GeomPolyline();
    GeomPolyline(const GeomPolyline&);
    virtual ~GeomPolyline();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SCICORESHARE GeomPolylineTC: public GeomObj {
protected:
    Array1<float> data;
    BBox bbox;
    int drawmode;
    double drawdist;
public:
    GeomPolylineTC(int drawmode, double drawdist);
    GeomPolylineTC(const GeomPolylineTC& copy);
    virtual ~GeomPolylineTC();

    void add(double t, const Point&, const Color&);
    
    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Polyline_h */
