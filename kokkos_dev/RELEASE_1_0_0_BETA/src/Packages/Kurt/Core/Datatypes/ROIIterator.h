#ifndef ROIITERATOR_H
#define ROIITERATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include "GLTextureIterator.h"
#include "GLTexture3D.h"

namespace Kurt {
using namespace SCIRun;


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
} // End namespace Kurt

#endif
