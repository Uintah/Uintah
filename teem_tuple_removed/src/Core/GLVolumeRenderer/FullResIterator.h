/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef FULLRESITERATOR_H
#define FULLRESITERATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/GLVolumeRenderer/GLTextureIterator.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>

namespace SCIRun {



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

} // End namespace SCIRun
#endif
