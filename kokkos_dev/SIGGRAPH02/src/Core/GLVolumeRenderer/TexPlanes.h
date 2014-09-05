/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef TEXPLANES_H
#define TEXPLANES_H

#include <Core/GLVolumeRenderer/GLVolRenState.h>

namespace SCIRun {

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

} // End namespace SCIRun

#endif
