#ifndef GLTEXTUREITERATOR_H
#define GLTEXTUREITERATOR_H

#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Point.h>
#include "GLTexture3D.h"
#include <vector>
#include <deque>


namespace Kurt {
using namespace SCIRun;
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
} // End namespace Kurt

#endif
