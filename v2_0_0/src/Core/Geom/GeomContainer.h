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
 *  GeomContainer.h: Base class for container objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Container_h
#define SCI_Geom_Container_h 1

#include <Core/Geom/GeomObj.h>

namespace SCIRun {

class SCICORESHARE GeomContainer : public GeomObj {
protected:
    GeomHandle child_;

public:
    GeomContainer(GeomHandle);
    GeomContainer(const GeomContainer&);
    virtual ~GeomContainer();
    virtual void get_bounds(BBox&);
    virtual void reset_bbox();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void get_triangles( Array1<float> &);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};    

} // End namespace SCIRun


#endif
