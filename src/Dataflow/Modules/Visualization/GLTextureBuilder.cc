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

#include <Core/Containers/Array1.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Datatypes/VolumeUtils.h>
#include <Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Network/Module.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {



static clString widget_name("GLTextureBuilderLocatorWidget");
static clString res_name("Resolution Widget");

extern "C" Module* make_GLTextureBuilder( const clString& id) {
  return scinew GLTextureBuilder(id);
}


GLTextureBuilder::GLTextureBuilder(const clString& id)
  : Module("GLTextureBuilder", id, Filter, "Visualization", "SCIRun"), 
    tex_(0),
    max_brick_dim_("max_brick_dim", id, this),
    min_("min", id, this),
    max_("max", id, this),
    is_fixed_("is_fixed", id, this),
    old_brick_size_(0), old_min_(-1), old_max_(-1)
{
  // Create the input ports
  infield_ = scinew FieldIPort( this, " Field", FieldIPort::Atomic);
  add_iport(infield_);

  // Create the output port
  otexture_ = scinew GLTexture3DOPort(this, "GL Texture", 
				      GLTexture3DIPort::Atomic);
  add_oport(otexture_);

}

GLTextureBuilder::~GLTextureBuilder()
{

}
/*
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
    cerr << "Unable to locate window!\n";
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return false;
  }
  std::cerr<<"here0"<<std::endl;
  dpy=Tk_Display(tkwin);
  std::cerr<<"here1"<<std::endl;
  win=Tk_WindowId(tkwin);
  std::cerr<<"here2"<<std::endl;
  vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributeList);

  cx=glXCreateContext(dpy, vi, 0, GL_TRUE);
  std::cerr<<"here3"<<std::endl;
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return false;
  }
  std::cerr<<"here4"<<std::endl;
  TCLTask::unlock();
  return true;
}

void
GLTextureBuilder::DestroyContext(Display *dpy, GLXContext& cx)
{
  glXDestroyContext(dpy,cx);
} 
*/
void GLTextureBuilder::execute(void)
{
  reset_vars();

  FieldHandle sfield;
  if (!infield_->get(sfield))
  {
    return;
  }
  else if (!sfield.get_rep() || sfield->get_type_name(0) != "LatticeVol")
  {
    return;
  }

  double minV, maxV;
  double min = min_.get();
  double max = max_.get();
  int is_fixed = is_fixed_.get();
  cerr << "is_fixed = "<<is_fixed<<"\n";
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

  old_min_ = minV;
  old_max_ = maxV;

  if (tex_.get_rep())
  {
    otexture_->send(tex_);
  }
}

} // End namespace SCIRun


