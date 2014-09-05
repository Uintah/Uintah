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
 *  GeomColorMap.h:
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
 *  Copyright (C) 2003  SCI Group
 */

#ifndef SCI_Geom_GeomColorMap_h 
#define SCI_Geom_GeomColorMap_h 1

#ifdef SCI_OPENGL
#include <Core/Geom/GeomOpenGL.h>
#endif
#include <Core/Geom/GeomContainer.h>
#include <Core/Geom/ColorMap.h>


namespace SCIRun {
    
class SCICORESHARE GeomColorMap : public GeomContainer
{
  ColorMapHandle cmap_;

public:
  GeomColorMap(GeomHandle geom, ColorMapHandle cmap);
  GeomColorMap(const GeomColorMap &copy);
      
  virtual GeomObj* clone();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;	
};
    
} // End namespace SCIRun

#endif
