#ifndef ROI_H
#define ROI_H

#include "GLVolRenState.h"

namespace SCICore {
namespace GeomSpace  {


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
  virtual void setAlpha(const Brick& brick);

};

}  // namespace GeomSpace
} // namespace SCICore

#endif
