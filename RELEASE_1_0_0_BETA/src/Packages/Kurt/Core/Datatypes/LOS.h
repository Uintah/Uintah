#ifndef LOS_H
#define LOS_H

#include "GLVolRenState.h"
namespace Kurt {

/**************************************

CLASS
   LOS
   
   LOS Class.

GENERAL INFORMATION

   LOS.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   LOS

DESCRIPTION
   LOS class.  Use subclasses to implement the drawing State
   for the VolumeRenderer.
  
WARNING
  
****************************************/

class LOS: public GLVolRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  LOS(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~LOS(){}

  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw();
  // draw Wireframe
  virtual void drawWireFrame();

protected:
  void setAlpha(const Brick& brick);
  void drawBrick(Brick& brick, const vector<Polygon *>& polys);
  

};
} // End namespace Kurt


#endif
