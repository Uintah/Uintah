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


#ifndef ROIITERATOR_H
#define ROIITERATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/GLVolumeRenderer/GLTextureIterator.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>

namespace SCIRun {



/**************************************

CLASS
   ROIIterator
   
   Region Of Influence IteratorClass.

GENERAL INFORMATION

   ROIIterator.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   Region Of Influence Iterator  class.  If we have a multiresolution texture,
   we may only want to resolve it in a certain region.  This iterator will
   step through the Bricks and return those in that region at full
   resolution.  Textures further away from the region will get
   rendered at lower resolution.
  
WARNING
  
****************************************/

class ROIIterator : public GLTextureIterator {
public:
  // GROUP: Constructors:
  //////////
  // Constructor
  ROIIterator(const GLTexture3D* tex, Ray view,
	      Point control);
  // GROUP: Destructors
  //////////
  // Destructor
  ~ROIIterator(){}
 
  // GROUP: Access
  //////////
  // get first brick
  virtual Brick* Start();
  //////////
  // get next brick
  virtual Brick* Next();
  // GROUP: Query
  //////////
  // are we finished?
  virtual bool isDone();

protected:
private:
  void SetNext();
};

} // End namespace SCIRun
#endif
