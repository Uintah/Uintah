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
 *  Sticky.h - Sticky frame for GeomObj's
 *
 *  Written by:
 *   Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   Jone 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_STICKY_H
#define SCI_STICKY_H 1

#include <Core/Geom/GeomContainer.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE GeomSticky : public GeomContainer {

public:
  GeomSticky( GeomHandle obj);
  GeomSticky(const GeomSticky&);
  virtual GeomObj* clone();

  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif
