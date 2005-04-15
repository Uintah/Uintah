#ifndef TEXPLANES_H
#define TEXPLANES_H

#include "GLVolRenState.h"

namespace SCICore {
  namespace Geometry {
    class Polygon;
  }
namespace GeomSpace  {
  using SCICore::Geometry::Polygon;

/**************************************

CLASS
   TexPlanes
   
   TexPlanes Class.

GENERAL INFORMATION

   TexPlanes.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TexPlanes

DESCRIPTION
   TexPlanes class.  Use subclasses to implement the drawing State
   for the VolumeRenderer.
  
WARNING
  
****************************************/

class TexPlanes : public GLVolRenState  {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  TexPlanes(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  ~TexPlanes(){}

  // GROUP: Operations
  //////////
  // draw the Volume
  virtual void draw();
  // draw Wireframe
  virtual void drawWireFrame();
  
protected:
  virtual void setAlpha(const Brick& brick);
  void draw(Brick&, Polygon*);

};

}  // namespace GeomSpace
} // namespace SCICore
#endif
