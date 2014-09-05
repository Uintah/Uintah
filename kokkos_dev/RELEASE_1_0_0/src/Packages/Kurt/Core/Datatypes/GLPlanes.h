#ifndef GLPLANES_H
#define GLPLANES_H

#include "GLTexRenState.h"

namespace Kurt {
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
} // End namespace Kurt

#endif
