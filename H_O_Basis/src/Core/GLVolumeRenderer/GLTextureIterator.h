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


#ifndef GLTEXTUREITERATOR_H
#define GLTEXTUREITERATOR_H

#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Point.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <deque>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {
  
  using std::vector;
  using std::deque;

/**************************************

CLASS
   GLTextureIterator
   
   GLTextureIterator Base Class.

GENERAL INFORMATION

   GLTextureIterator.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   GLTextureIterator Base class.
  
WARNING
  
****************************************/

class GLTextureIterator {
public:
  // GROUP: Constructors:
  //////////
  // Constructor
  GLTextureIterator(const GLTexture3D* tex, Ray view,
		    Point control);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLTextureIterator();
 
  // GROUP: Access
  //////////
  // get first brick
  virtual Brick* Start() = 0;
  //////////
  // get next brick
  virtual Brick* Next() = 0;
  // GROUP: Query
  //////////
  // are we finished?
  virtual bool isDone() = 0;

protected:
  Ray view;
  Point control;
  const GLTexture3D* tex;
  bool done;
  Brick* next;

  vector< const Octree<Brick*>* >  path;
  vector< deque<int>* > order;
  
  static int traversalTable[27][8];

  deque<int>* traversal(const Octree<Brick*>* node);
  
private:
  GLTextureIterator(){}
};

} // End namespace SCIRun
#endif
