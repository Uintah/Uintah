#ifndef GLVOLRENSTATE_H
#define GLVOLRENSTATE_H

//#include "GLVolumeRenderer.h"
#include <SCICore/Geometry/Ray.h>
#include <vector>
#include <GL/glu.h>

namespace Kurt {
  namespace Datatypes {
  class Brick;
  }
}

namespace SCICore {
  namespace Geometry {
  class Polygon;
  }
  namespace GeomSpace {
  
  //using namespace Kurt::Datatypes;
  using SCICore::Geometry::Ray;
  using SCICore::Geometry::Polygon;
  using std::vector;
  using namespace Kurt::Datatypes;
  
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

  virtual void setAlpha(const Brick& brick) = 0;
  
  void computeView(Ray&);
  void loadTexture( Brick& brick);
  void makeTextureMatrix(const Brick& brick);
  void enableTexCoords();
  void drawPolys( vector<Polygon *> polys);
  void disableTexCoords();
  void drawWireFrame(const Brick& brick);
  const GLVolumeRenderer*  volren;

  GLuint* texName;
  unsigned char* reload;
};

} // end namespace GeomSpace
} // end namespace SCICore
#endif
