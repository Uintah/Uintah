#ifndef FULLRESITERATOR_H
#define FULLRESITERATOR_H

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Datatypes/GLTextureIterator.h>
#include <SCICore/Datatypes/GLTexture3D.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Geometry::Point;
using SCICore::Geometry::Ray;


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

} // end namespace Datatypes
} // end namespace SCICore
#endif
