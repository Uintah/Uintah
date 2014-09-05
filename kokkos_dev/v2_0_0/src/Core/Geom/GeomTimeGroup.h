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
 *  GeomTimeGroup.h:  ?
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Time__Group_h
#define SCI_Geom_Time_Group_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomTimeGroup : public GeomObj {
    vector<GeomHandle> objs;
    vector<double>     start_times;

    BBox bbox; // bbox for entire seen - set once!
public:
    GeomTimeGroup();
    GeomTimeGroup(const GeomTimeGroup&);
    virtual GeomObj* clone();

    void add(GeomHandle,double); // with time...
    void remove(GeomHandle);
    void remove_all();
    int size();

    void  setbbox(BBox&); // sets bounding box - so isn't computed!

    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Group_h */
