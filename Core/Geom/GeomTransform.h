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
 *  Transform.h:  Transform Properities for Geometry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Geom_Transform_h
#define SCI_Geom_Transform_h 1

#include <Core/Persistent/Persistent.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geom/GeomContainer.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {


class SCICORESHARE GeomTransform : public GeomContainer {
    Transform trans;
public:
    GeomTransform(GeomHandle);
    GeomTransform(GeomHandle, const Transform);
    GeomTransform(const GeomTransform&);
    void setTransform(const Transform);
    Transform getTransform();
    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

    void scale(const Vector&);
    void translate(const Vector&);
    void rotate(double, const Vector&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif
