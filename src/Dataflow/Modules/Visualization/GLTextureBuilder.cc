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

#include <Core/Datatypes/LatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Network/Module.h>

#include <iostream>

namespace SCIRun {



static string widget_name("GLTextureBuilderLocatorWidget");
static string res_name("Resolution Widget");

extern "C" Module* make_GLTextureBuilder( const string& id) {
  return scinew GLTextureBuilder(id);
}


GLTextureBuilder::GLTextureBuilder(const string& id)
  : Module("GLTextureBuilder", id, Filter, "Visualization", "SCIRun"), 
    tex_(0),
    is_fixed_("is_fixed", id, this),
    max_brick_dim_("max_brick_dim", id, this),
    min_("min", id, this),
    max_("max", id, this),
    old_brick_size_(0), old_min_(-1), old_max_(-1)
{
}

GLTextureBuilder::~GLTextureBuilder()
{

}

#if 0
bool
GLTextureBuilder::MakeContext(Display *dpy, GLXContext& cx)
{
  Tk_Window tkwin;
  Window win;
  XVisualInfo *vi;
  
  TCLTask::lock();
  char *myn=strdup(".");
  tkwin=Tk_NameToWindow(the_interp, myn, Tk_MainWindow(the_interp));
  if(!tkwin){
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return false;
  }
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributeList);

  cx=glXCreateContext(dpy, vi, 0, GL_TRUE);
  if(!cx){
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return false;
  }
  TCLTask::unlock();
  return true;
}

void
GLTextureBuilder::DestroyContext(Display *dpy, GLXContext& cx)
{
  glXDestroyContext(dpy,cx);
} 
#endif

void GLTextureBuilder::execute(void)
{
  infield_ = (FieldIPort *)get_iport("Field");
  otexture_ = (GLTexture3DOPort *)get_oport("GL Texture");
  FieldHandle sfield;

  if (!infield_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!otexture_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
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
  if (!sfield.get_rep() ||
      (sfield->mesh()->get_type_description()->get_name() !=
       get_type_description((LatVolMesh *)0)->get_name()))
  {
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
    TCL::execute(id + " SetDims " + to_string( tex_->get_brick_size()));
    max_brick_dim_.set(tex_->get_brick_size());
    old_brick_size_ = tex_->get_brick_size();
  }
  else if (sfield.get_rep() != sfrg_.get_rep())
  {
    sfrg_ = sfield;
    if (is_fixed) {
      minV = min;
      maxV = max;
    }

    // see note above
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);
    if (!is_fixed) {
      tex_->getminmax(minV, maxV);
      min_.set(minV);
      max_.set(maxV);
    }
    tex_->set_brick_size(max_brick_dim_.get());
  }
  else if (old_brick_size_ != max_brick_dim_.get())
  {
    tex_->set_brick_size(max_brick_dim_.get());
    old_brick_size_ = max_brick_dim_.get();
  }
  else if ((old_min_ != min) || (old_max_ != max))
  {
    if (is_fixed) {
      minV = min;
      maxV = max;
    }

    // see note above
    tex_ = scinew GLTexture3D(sfield, minV, maxV, is_fixed);
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


