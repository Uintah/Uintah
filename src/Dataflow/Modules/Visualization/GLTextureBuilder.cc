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
    tex(0),
    max_brick_dim("max_brick_dim", id, this),
//    isFixed("isFixed", id, this),
    min("min", id, this),
    max("max", id, this)
{
  // Create the input ports
  infield = scinew FieldIPort( this, " Field",
					   FieldIPort::Atomic);
  add_iport(infield);

  // Create the output port
  otexture = scinew GLTexture3DOPort(this, "GL Texture", 
				     GLTexture3DIPort::Atomic);
  add_oport(otexture);

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
  //  Display *dpy;
  // GLXContext cx;
  //static bool init = false;
  FieldHandle sfield;
  static int oldBrickSize = 0;
  if (!infield->get(sfield)) {
    return;
  } else if (sfield->get_type_name(0) != "LatticeVol") {
    return;
  }

  double minV, maxV;
  if( sfield.get_rep() != sfrg.get_rep()  && !tex.get_rep() ){
    sfrg = sfield;
    tex = scinew GLTexture3D(sfield);
    TCL::execute(id + " SetDims " + to_string( tex->get_brick_size()));
    max_brick_dim.set( tex->get_brick_size() );
    oldBrickSize = tex->get_brick_size();
    tex->getminmax(minV, maxV);
    min.set(minV); max.set(maxV);
  } else if( sfield.get_rep() != sfrg.get_rep()){
    sfrg = sfield;
    tex = scinew GLTexture3D(sfield);
    tex->set_brick_size( max_brick_dim.get() );
    tex->getminmax(minV, maxV);
    min.set(minV); max.set(maxV);
  } else if (oldBrickSize != max_brick_dim.get()){
    tex->set_brick_size( max_brick_dim.get() );
    oldBrickSize = max_brick_dim.get();
  }
  
  if( tex.get_rep() )
    otexture->send( tex );
}

} // End namespace SCIRun


