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

#ifndef GLPLANES_H
#define GLPLANES_H

#include <Core/GLVolumeRenderer/GLTexRenState.h>

namespace SCIRun {

/**************************************

CLASS
   GLPlanes
   
   GLPlanes Class.

GENERAL INFORMATION

   GLPlanes.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLPlanes

DESCRIPTION
   GLPlanes class.  A GLdrawing State
   for the GLVolumeRenderer.
  
WARNING
  
****************************************/

class GLPlanes : public GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLPlanes(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLPlanes(){}
  // GROUP: Operations
  //////////
  // predrawing gl functions
  virtual void preDraw();
  //////////
  // postdrawing functions
  virtual void postDraw();
  //////////
private:

};

} // End namespace SCIRun
#endif
