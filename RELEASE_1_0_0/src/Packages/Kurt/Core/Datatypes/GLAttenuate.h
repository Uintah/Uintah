#ifndef GL_ATTENUATE_H
#define GL_ATTENUATE_H

#include "GLTexRenState.h"
namespace Kurt {
/**************************************

CLASS
   GLAttenuate
   
   GLAttenuate Class.

GENERAL INFORMATION

   GLAttenuate.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLAttenuate

DESCRIPTION
   GLAttenuate class.  A GLdrawing State
   for the GLVolumeRenderer.
  
WARNING
  
****************************************/

class GLAttenuate : public GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLAttenuate(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLAttenuate(){}
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
