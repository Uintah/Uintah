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



/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Modules/Visualization/GLTextureBuilder.h>
#include <Dataflow/Network/Module.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Malloc/Allocator.h>

#include <sys/types.h>
#include <unistd.h>
#include <iostream>

namespace SCIRun {



static string widget_name("GLTextureBuilderLocatorWidget");
static string res_name("Resolution Widget");

DECLARE_MAKER(GLTextureBuilder)

GLTextureBuilder::GLTextureBuilder(GuiContext* ctx) : 
  Module("GLTextureBuilder", ctx, Filter, "Visualization", "SCIRun"), 
  tex_(0),
  is_fixed_(ctx->subVar("is_fixed")),
  max_brick_dim_(ctx->subVar("max_brick_dim")),
  sel_brick_dim_(ctx->subVar("sel_brick_dim")),
  min_(ctx->subVar("min")),
  max_(ctx->subVar("max")),
  old_brick_size_(0), 
  old_min_(-1), 
  old_max_(-1)
{
}

GLTextureBuilder::~GLTextureBuilder()
{

}

void GLTextureBuilder::execute(void)
{
  infield_ = (FieldIPort *)get_iport("Field");
  otexture_ = (GLTexture3DOPort *)get_oport("GL Texture");
  FieldHandle sfield;

  if (!infield_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!otexture_) {
    error("Unable to initialize oport 'GL Texture'.");
    return;
  } 
  if (!infield_->get(sfield))
  {
    return;
  }
  real_execute(sfield);
}

void GLTextureBuilder::real_execute(FieldHandle sfield)
{
  //char me[] = "GLTextureBuilder::real_execute:";
  
  if (!sfield.get_rep() ||
      (sfield->mesh()->get_type_description()->get_name() !=
       get_type_description((LatVolMesh *)0)->get_name()))
  {
    error("Input field has no representation or is not a LatVolMesh.");
    return;
  }

  if (!sfield->query_scalar_interface(this).get_rep())
  {
    error("Input field is of nonscalar LatVolField type.");
    return;
  }

  reset_vars();

  double minV, maxV;
  double min = min_.get();
  double max = max_.get();
  int is_fixed = is_fixed_.get();

  if (sfield.get_rep() != sfrg_.get_rep()  && !tex_.get_rep())
  {
    sfrg_ = sfield;
    if (is_fixed) {  // if fixed, copy min/max into these locals
      minV = min;
      maxV = max;
    }

    // this constructor will take in a min and max, and if is_fixed is set
    // it will set the values to that range... otherwise it auto-scales
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);
    
    if (!is_fixed) { // if not fixed, overwrite min/max values on Gui
      tex_->getminmax(minV, maxV);
      min_.set(minV);
      max_.set(maxV);
    }
    gui->execute(id + " SetDims " + to_string( tex_->get_brick_size()));
    if (sel_brick_dim_.get()) tex_->set_brick_size(sel_brick_dim_.get());
    old_brick_size_ = tex_->get_brick_size();
  }
  else if (sfield.get_rep() != sfrg_.get_rep())
  {
    sfrg_ = sfield;
    if (is_fixed) {
      minV = min;
      maxV = max;
    }

    // The boolean result of GLTexture3D::replace_data is whether or not the
    // data structure was able to be reused.  It has nothing to do with the
    // actuall values in the field (i.e. if they changed or not.  Therefore
    // we need to check to see if the min and max values of the field have
    // changed.
    if( !tex_->replace_data(sfield, minV, maxV, is_fixed) ){
      // see note above
      tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);

      tex_->set_brick_size(sel_brick_dim_.get());
      old_brick_size_ = sel_brick_dim_.get();
    }

    if (!is_fixed) {
      tex_->getminmax(minV, maxV);
      min_.set(minV);
      max_.set(maxV);
    }
  }
  else if (old_brick_size_ != sel_brick_dim_.get())
  {
    tex_->set_brick_size(sel_brick_dim_.get());
    old_brick_size_ = sel_brick_dim_.get();
  }
  else if ((old_min_ != min) || (old_max_ != max))
  {
    if (is_fixed) {
      minV = min;
      maxV = max;
    }

    if( !tex_->replace_data(sfield, minV, maxV, is_fixed) ){
      // see note above
      tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);
    }    

    if (!is_fixed) {
      tex_->getminmax(minV, maxV);
      min_.set(minV);
      max_.set(maxV);
    }
  }

  old_min_ = (int)minV;
  old_max_ = (int)maxV;

  if (tex_.get_rep())
  {
    otexture_->send(tex_);
  }
}

} // End namespace SCIRun







