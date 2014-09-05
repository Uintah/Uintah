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

#ifndef UINTAH_GLTEXTURE3D_H
#define UINTAH_GLTEXTURE3D_H

#include <Core/Datatypes/GLTexture3D.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Field.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
namespace Uintah {

using SCIRun::FieldHandle;

/**************************************

CLASS
   GLTexture3D
   
   Simple GLTexture3D Class.

GENERAL INFORMATION

   GLTexture3D.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   GLTexture3D class.
  
WARNING
  
****************************************/
class GLTexture3D : public SCIRun::GLTexture3D {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  GLTexture3D(FieldHandle texfld, double &min, double &max, int use_minmax);
  //////////
  // Constructor
  GLTexture3D();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GLTexture3D();
  virtual void set_field(FieldHandle tex);
  virtual bool set_brick_size( int brickSize );
  virtual bool get_dimensions( int& nx, int& ny, int& nz);
protected:
  virtual void build_texture();


};


} // End namespace Uintah



#endif //UINTAH_GLTEXTURE3D_H
