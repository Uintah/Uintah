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
#include <Core/GLVolumeRenderer/GLTexture3D.h>
#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Core/Datatypes/Field.h>
#include <GL/glx.h>


namespace SCIRun {
  class GLTexture3D;
}

namespace SCIRun {



class GLTextureBuilder : public Module {

public:
  GLTextureBuilder(GuiContext* ctx);

  virtual ~GLTextureBuilder();

  virtual void execute();
  void real_execute(FieldHandle fh);
protected:
  FieldIPort *infield_;

  GLTexture3DOPort* otexture_;
   
  FieldHandle sfrg_;
  GLTexture3DHandle tex_;

  GuiInt is_fixed_;
  GuiInt max_brick_dim_;
  GuiInt sel_brick_dim_;
  GuiDouble min_, max_;
  int old_brick_size_;
  int old_min_;
  int old_max_;
};

} // End namespace SCIRun

#endif
