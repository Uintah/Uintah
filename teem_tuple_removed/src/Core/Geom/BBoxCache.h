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
 *  BBoxCache.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_Geom_BBoxCache_h 
#define SCI_Geom_BBoxCache_h 1

#include <Core/Geom/GeomContainer.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomBBoxCache: public GeomContainer {
  
    bool bbox_cached;
    BBox bbox;

public:
    GeomBBoxCache(GeomHandle);
    GeomBBoxCache(GeomHandle, const BBox &);

    virtual GeomObj* clone();
    virtual void reset_bbox();
    virtual void get_bounds(BBox&);
    
    virtual void io(Piostream&);
    static PersistentTypeID type_id;	
};

} // End namespace SCIRun

#endif
