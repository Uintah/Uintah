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
 *  TimeGrid.h: ?
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_TimeGeom_Grid_h
#define SCI_TimeGeom_Grid_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array2.h>

class GeomColormapInterface;

namespace SCIRun {

class SCICORESHARE TimeGrid : public GeomObj {
    Array1<float *> tmap;

    float *bmap; // blend map...

    int dimU,dimV;
    int tmap_size; // all the same size!!!

    Array1<int>    tmap_dlist;
    Array1<double> time;

    Point corner;
    Vector u, v, w; // all the same!!!
    int active;
    void adjust();
public:
    TimeGrid(int,int, int, const Point&, const Vector&, const Vector&);
    TimeGrid(const TimeGrid&);
    virtual ~TimeGrid();

    virtual GeomObj* clone();

    // methor for textur mapping...

    GeomColormapInterface* map; // color map to be used...

    void set_active(int, double);

    void set(int,int, const MaterialHandle&, const double&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Grid_h */
