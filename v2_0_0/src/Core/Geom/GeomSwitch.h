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
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Switch_h
#define SCI_Geom_Switch_h 1

#include <Core/Geom/GeomContainer.h>

namespace SCIRun {

class SCICORESHARE GeomSwitch : public GeomContainer {
    int state;
    GeomSwitch(const GeomSwitch&);

public:
    GeomSwitch(GeomHandle, int state=1);
    virtual GeomObj* clone();

    void set_state(int st);
    int get_state();
    virtual void get_bounds(BBox&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class SCICORESHARE GeomTimeSwitch : public GeomContainer {
    double tbeg;
    double tend;
    GeomTimeSwitch(const GeomTimeSwitch&);

public:
    GeomTimeSwitch(GeomHandle, double tbeg, double tend);
    virtual GeomObj* clone();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


} // End namespace SCIRun


#endif /* SCI_Geom_Switch_h */
