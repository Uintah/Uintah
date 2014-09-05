#ifndef ROI_H
#define ROI_H

#include "GLVolRenState.h"

namespace Kurt {

/**************************************

CLASS
   ROI
   
   ROI Class.

GENERAL INFORMATION

   ROI.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ROI

DESCRIPTION
   ROI class.  Use subclasses to implement the drawing State
   for the VolumeRenderer.
  
WARNING
  
****************************************/

class ROI: public GLVolRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  ROI(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~ROI(){}

  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw();
  // draw Wireframe
  virtual void drawWireFrame();



protected:
  void setAlpha(const Brick& brick);
  void drawBrick( Brick& brick, const vector<Polygon *>& polys);

};
} // End namespace Kurt


#endif
