#ifndef GLTEXRENSTATE_H
#define GLTEXRENSTATE_H

namespace Kurt {

/**************************************

CLASS
   GLTexRenState
   
   GLTexRenState Class.

GENERAL INFORMATION

   GLTexRenState.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLTexRenState

DESCRIPTION
   GLTexRenState class.  Use subclasses to implement the GLdrawing State
   for the GLTexureRenderer.
  
WARNING
  
****************************************/
class GLVolumeRenderer;

class GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLTexRenState(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLTexRenState(){}
  // GROUP: Operations
  //////////
  // predrawing gl functions
  virtual void preDraw() = 0;
  //////////
  // postdrawing functions
  virtual void postDraw() = 0;

protected:

  const GLVolumeRenderer* volren;
};
} // End namespace Kurt

#endif
