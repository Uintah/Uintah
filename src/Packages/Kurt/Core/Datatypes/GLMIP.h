#ifndef GLMIP_H
#define GLMIP_H

#include "GLTexRenState.h"

namespace Kurt {
/**************************************

CLASS
   GLMIP
   
   GLMIP Class.

GENERAL INFORMATION

   GLMIP.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLMIP

DESCRIPTION
   GLMIP class.  A GLdrawing State
   for the GLVolumeRenderer.
  
WARNING
  
****************************************/

class GLMIP : public GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLMIP(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLMIP(){}
  // GROUP: Operations
  //////////
  // predrawing gl functions
  virtual void preDraw();
  //////////
  // postdrawing functions
  virtual void postDraw();

private:

};
} // End namespace Kurt

#endif
