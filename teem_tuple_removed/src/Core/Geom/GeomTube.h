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
 *  GeomTube.h: Tube object
 *
 *  Written by:
 *   Han-Wei Shen
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tube_h 
#define SCI_Geom_Tube_h 1 

#include <Core/Geom/GeomVertexPrim.h>

class SinCosTable;

namespace SCIRun {

class SCICORESHARE GeomTube : public GeomVertexPrim {
    int nu;
    Array1<Vector> directions;
    Array1<double> radii;
private:
    void make_circle(int which, Array1<Point>& circle,
		     const SinCosTable& tab); 
public:
    GeomTube(int nu=8); 
    GeomTube(const GeomTube&); 
    virtual ~GeomTube(); 

    virtual GeomObj* clone(); 
    virtual void get_bounds(BBox&); 
  
    void add(GeomVertex*, double, const Vector&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time); 
#endif 

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun



#endif /*SCI_Geom_Tube_h */
