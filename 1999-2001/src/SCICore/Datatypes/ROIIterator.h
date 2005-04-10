#ifndef ROIITERATOR_H
#define ROIITERATOR_H

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

} // end namespace Datatypes
} // end namespace SCICore
#endif
