
/*
 * GLTextureBuilder.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "GLTextureBuilder.h"
#include <sys/types.h>
#include <unistd.h>



#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarField.h>


#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLTask.h>

#include <Kurt/Datatypes/VolumeUtils.h>
#include <Kurt/Datatypes/GLTexture3D.h>

//#include <GL/gl.h>
//#include <GL/glx.h>
//#include <tcl.h>
//#include <tk.h>
#include <iostream>
using std::cerr;
using std::endl;

// extern Tcl_Interp* the_interp;
// extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
// static int attributeList[] = { GLX_RGBA, None };

namespace Kurt {
namespace Modules {

using namespace SCICore::TclInterface;
using namespace SCICore::Math;

using namespace SCICore::Containers;
using namespace SCICore::TclInterface;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Datatypes::ScalarFieldHandle;
using SCICore::Datatypes::ScalarFieldRGBase;
using PSECore::Datatypes::GLTexture3DOPort;
using PSECore::Datatypes::GLTexture3DIPort;
using PSECore::Datatypes::ScalarFieldOPort;

using Kurt::Datatypes::GLTexture3D;
using Kurt::Datatypes::GLTexture3DHandle;

static clString widget_name("GLTextureBuilderLocatorWidget");
static clString res_name("Resolution Widget");
			 
extern "C" Module* make_GLTextureBuilder( const clString& id) {
  return scinew GLTextureBuilder(id);
}


GLTextureBuilder::GLTextureBuilder(const clString& id)
  : Module("GLTextureBuilder", id, Filter), 
    max_brick_dim("max_brick_dim", id, this),
    min("min", id, this),
    max("max", id, this),
    isFixed("isFixed", id, this),
    tex(0) 
{
  // Create the input ports
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					   ScalarFieldIPort::Atomic);
  add_iport(inscalarfield);

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
  static bool init = false;
  ScalarFieldHandle sfield;
  static int oldBrickSize = 0;
  if (!inscalarfield->get(sfield)) {
    return;
  }
  else if (!sfield.get_rep()) {
    return;
  }
  
  if( isFixed.get() ){
    double min;
    double max;
    sfield->get_minmax(min, max);
    if (min != this->min.get() || max != this->max.get())
      sfrg = 0;
    sfield->set_minmax(this->min.get(), this->max.get());
  } else {
      double min;
      double max;
      sfield->get_minmax(min, max);
      this->min.set( min );
      this->max.set( max );
    }

  if( ScalarFieldRGBase *rg =
      dynamic_cast<ScalarFieldRGBase *> (sfield.get_rep()) ){
    
    if( sfield.get_rep() != sfrg.get_rep()  && !tex.get_rep() ){
      sfrg = sfield;
      tex = scinew GLTexture3D( rg);
      TCL::execute(id + " SetDims " + to_string( tex->getBrickSize()));
      max_brick_dim.set( tex->getBrickSize() );
      oldBrickSize = tex->getBrickSize();
    } else if( sfield.get_rep() != sfrg.get_rep()){
      sfrg = sfield;
      tex = scinew GLTexture3D( rg);
      tex->SetBrickSize( max_brick_dim.get() );
    } else if (oldBrickSize != max_brick_dim.get()){
      tex->SetBrickSize( max_brick_dim.get() );
      oldBrickSize = max_brick_dim.get();
    }

    if( tex.get_rep() )
      otexture->send( tex );
    
  } else {
    cerr << "Not an rg field!\n";
    otexture->send( 0 );
  }
}

} // End namespace Modules
} // End namespace Uintah


