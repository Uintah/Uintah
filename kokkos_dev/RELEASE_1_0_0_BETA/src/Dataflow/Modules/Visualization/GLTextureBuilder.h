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

#ifndef GLTEXTUREBUILDER_H
#define GLTEXTUREBUILDER_H
/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Core/Datatypes/Field.h>
#include <GL/glx.h>


namespace SCIRun {
  class GLTexture3D;
}

namespace SCIRun {



class GLTextureBuilder : public Module {

public:
  GLTextureBuilder( const clString& id);

  virtual ~GLTextureBuilder();

  virtual void execute();

private:
  FieldIPort *infield;

  GLTexture3DOPort* otexture;
   
  Point Smin,Smax;
  Vector ddv;
  
  FieldHandle sfrg;
  GLTexture3DHandle tex;

  GuiInt max_brick_dim;
  GuiDouble min, max;
  GuiInt isFixed;
//  bool MakeContext(Display *dpy, GLXContext& cx);
  // void DestroyContext(Display *dpy, GLXContext& cx);

};

} // End namespace SCIRun

#endif
