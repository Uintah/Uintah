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


/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "GLTextureBuilder.h"
#include <sys/types.h>
#include <unistd.h>

#include <Core/Datatypes/LatVolField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>

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







