/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
