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
 *  Tri.h: Triangles...
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tri_h
#define SCI_Geom_Tri_h 1

#include <Core/Geom/GeomVertexPrim.h>

namespace SCIRun {

class SCICORESHARE GeomTri : public GeomVertexPrim {
    Vector n;
public:
    GeomTri(const Point&, const Point&, const Point&);
    GeomTri(const Point&, const Point&, const Point&,
	    const MaterialHandle&,
	    const MaterialHandle&,
	    const MaterialHandle&);
    GeomTri(const GeomTri&);
    virtual ~GeomTri();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Tri_h */
