/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
