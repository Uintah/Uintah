#ifndef LOS_H
#define LOS_H

#include <SCICore/Datatypes/GLVolRenState.h>

namespace SCICore {
namespace GeomSpace  {

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
  virtual void setAlpha(const Brick& brick);
  

};

}  // namespace GeomSpace
} // namespace SCICore

#endif
