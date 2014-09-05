#ifndef GLVOLRENSTATE_H
#define GLVOLRENSTATE_H

#include "Polygon.h"
#include "Brick.h"
#include <vector>

class GLVolumeRenderer;

namespace Kurt {
  using std::vector;
/**************************************

CLASS
   GLVolRenState
   
   GLVolRenState Class.

GENERAL INFORMATION

   GLVolRenState.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLVolRenState

DESCRIPTION
   GLVolRenState class.  Use subclasses to implement the drawing State
   for the VolumeRenderer.
  
WARNING
  
****************************************/

class GLVolRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLVolRenState(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~GLVolRenState(){}
  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw() = 0;
  // draw Wireframe
  virtual void drawWireFrame() = 0;

protected:

  Ray& computeView();
  void loadTexture( const Brick& brick);
  void makeTextureMatrix();
  void enableTexCoords();
  void drawPolys( vector<Polygon *> polys);
  void disableTexCoords();
  void drawWireFrame(const Brick& brick);
  const GLVolumeRenderer*  volren;
};
} // End namespace Kurt

#endif
