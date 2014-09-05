#ifndef GLVOLRENSTATE_H
#define GLVOLRENSTATE_H

//#include "GLVolumeRenderer.h"
#include <Core/Geometry/Ray.h>
#include <vector>
#include <GL/glu.h>

namespace Kurt {

using namespace SCIRun;
using std::vector;

class Brick;
class Polygon;

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
class GLVolumeRenderer;

class GLVolRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLVolRenState(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLVolRenState(){}
  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw() = 0;
  // draw Wireframe
  virtual void drawWireFrame() = 0;
  
  void Reload(){reload = (unsigned char *)1;}

  
protected:

  void computeView(Ray&);
  void loadColorMap( Brick& brick );
  void loadTexture( Brick& brick);
  void makeTextureMatrix(const Brick& brick);
  void enableTexCoords();
  void enableBlend();
  void drawPolys( const vector<Polygon *>& polys);
  void disableTexCoords();
  void disableBlend();
  void drawWireFrame(const Brick& brick);
  void drawWirePolys( const vector<Polygon *>& polys );
  const GLVolumeRenderer*  volren;

  GLuint* texName;
  unsigned char* reload;

};

} // End namespace Kurt


#endif
