#ifndef FULLRESITERATOR_H
#define FULLRESITERATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include "GLTextureIterator.h"
#include "GLTexture3D.h"

namespace Kurt {
using namespace SCIRun;


/**************************************

CLASS
   FullResIterator
   
   FullResIterator Base Class.

GENERAL INFORMATION

   FullResIterator.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   FullResIterator Base class.
  
WARNING
  
****************************************/

class FullResIterator : public GLTextureIterator {
public:
  // GROUP: Constructors:
  //////////
  // Constructor
  FullResIterator(const GLTexture3D* tex, Ray view,
		  Point control);
  // GROUP: Destructors
  //////////
  // Destructor
  ~FullResIterator(){}
 
  // GROUP: Access
  //////////
  // get first brick
  virtual  Brick* Start();
  //////////
  // get next brick
  virtual Brick* Next();
  // GROUP: Query
  //////////
  // are we finished?
  virtual bool isDone();

protected:
  void SetNext();
private:
};
} // End namespace Kurt

#endif
