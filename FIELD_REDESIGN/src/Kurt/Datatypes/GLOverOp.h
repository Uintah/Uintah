#ifndef GLOVEROP_H
#define GLOVEROP_H
#include "GLTexRenState.h"

namespace SCICore {
namespace GeomSpace  {

/**************************************

CLASS
   GLOverOp
   
   GLOverOp Class.

GENERAL INFORMATION

   GLOverOp.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLOverOp

DESCRIPTION
   GLOverOp class.  A GLdrawing State
   for the GLVolumeRenderer.
  
WARNING
  
****************************************/

class GLOverOp : public GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLOverOp(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLOverOp(){}
  // GROUP: Operations
  //////////
  // predrawing gl functions
  virtual void preDraw();
  //////////
  // postdrawing functions
  virtual void postDraw();
  //////////
  // 
  static GLTexRenState* Instance(const GLVolumeRenderer* glvr);
  //////////
  // 
  //////////
  // 
private:
  static GLTexRenState* _instance;

};

} // end namespace Datatypes
} // end namespace Kurt
#endif
