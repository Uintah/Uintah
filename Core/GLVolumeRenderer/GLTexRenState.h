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


#ifndef GLTEXRENSTATE_H
#define GLTEXRENSTATE_H

namespace SCIRun {


/**************************************

CLASS
   GLTexRenState
   
   GLTexRenState Class.

GENERAL INFORMATION

   GLTexRenState.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   GLTexRenState

DESCRIPTION
   GLTexRenState class.  Use subclasses to implement the GLdrawing State
   for the GLTexureRenderer.
  
WARNING
  
****************************************/
class GLVolumeRenderer;


class GLTexRenState {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  GLTexRenState(const GLVolumeRenderer* glvr);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLTexRenState(){}
  // GROUP: Operations
  //////////
  // predrawing gl functions
  virtual void preDraw() = 0;
  //////////
  // postdrawing functions
  virtual void postDraw() = 0;
  //////////
  //////////
  //////////
protected:

  const GLVolumeRenderer* volren;
};



} // End namespace SCIRun

#endif
