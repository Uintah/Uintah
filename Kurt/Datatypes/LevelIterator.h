#ifndef LEVELITERATOR_H
#define LEVELITERATOR_H

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include "GLTextureIterator.h"
#include "GLTexture3D.h"

namespace Kurt {
namespace Datatypes {

using SCICore::Geometry::Point;
using SCICore::Geometry::Ray;


/**************************************

CLASS
   LevelIterator
   
   LevelIterator Base Class.

GENERAL INFORMATION

   LevelIterator.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   LevelIterator Base class.
  
WARNING
  
****************************************/

class LevelIterator : public GLTextureIterator {
public:
  // GROUP: Constructors:
  //////////
  // Constructor
  LevelIterator(const GLTexture3D* tex, Ray view,
		  Point control, int level);
  // GROUP: Destructors
  //////////
  // Destructor
  ~LevelIterator(){}
 
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
  int level;
  void SetNext();
private:
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
