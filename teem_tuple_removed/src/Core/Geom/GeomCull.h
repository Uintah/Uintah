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
 *  GenAxes.cc: Choose one input field to be passed downstream
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   November 12 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef SCI_Geom_Cull_h
#define SCI_Geom_Cull_h 1

#include <Core/Geom/GeomContainer.h>

namespace SCIRun {

class SCICORESHARE GeomCull : public GeomContainer {
  Vector *normal_;
  GeomCull(const GeomCull &copy);

  public:
  GeomCull(GeomHandle, Vector *);
  virtual GeomObj* clone(); 
  
  void set_normal(Vector *);
    
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&stream);
  static PersistentTypeID type_id;  
};

}

#endif
