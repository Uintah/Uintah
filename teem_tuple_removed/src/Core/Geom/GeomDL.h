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
 *  GeomDL.h: ?
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Date July 2000
 *
 *  Copyright (C) 2000  SCI Group
 */

#ifndef SCI_Geom_GeomDL_h 
#define SCI_Geom_GeomDL_h 1

#ifdef SCI_OPENGL
#include <Core/Geom/GeomOpenGL.h>
#endif
#include <Core/Geom/GeomContainer.h>

#include <list>

namespace SCIRun {
    
class DrawInfoOpenGL;
using std::list;

class SCICORESHARE GeomDL : public GeomContainer {
protected:
  int polygons_;
  list<DrawInfoOpenGL *> drawinfo_;

public:
  GeomDL(GeomHandle);
  GeomDL(const GeomDL &copy);
  virtual ~GeomDL();
      
  virtual GeomObj* clone();
  virtual void reset_bbox();

  void dl_register(DrawInfoOpenGL *info);
  void dl_unregister(DrawInfoOpenGL *info);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;	
};
    
} // End namespace SCIRun

#endif
