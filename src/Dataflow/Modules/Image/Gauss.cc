//static char *id="@(#) $Id"

/*
 *  Gauss.cc:  
 *
 *  Written by:
 *    Scott Morris
 *    August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Containers/Array1.h>
#include <Util/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldRGshort.h>
#include <Datatypes/ScalarFieldRGfloat.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <TclInterface/TCLvar.h>
#include <TclInterface/TCLTask.h>

#include <GL/gl.h>
#include <GL/glu.h>

#include <Geom/GeomOpenGL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

// tcl interpreter corresponding to this module

extern Tcl_Interp* the_interp;

// the OpenGL context structure

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;

class Gauss : public Module {
  ScalarFieldIPort *inscalarfield;
  ScalarFieldOPort *outscalarfield;
  
  int grid_id;

  ScalarFieldRG* ingrid;
  ScalarFieldRG* outgrid;
  
  ScalarFieldRGfloat* dgrid;
  
  ScalarField* sfield;

  ScalarFieldHandle handle;

  GLXContext		ctx;    // OpenGL Contexts
  Display		*dpy;
  GLXPbufferSGIX        pbuf;
  Window		win;

  int                   winX,winY; // size of window in pixels
  int                   bdown;
  int                   width,height,oldwidth,oldheight;
  int                   drawn,rgb,m1,m2;
  int gen;

  double sig;
  int siz;
  int np;
  float *gauss;
  
  TCLdouble sigma,size;
  TCLint hardware;
  
public: 
  Gauss(const clString& id);
  Gauss(const Gauss&, int deep);
  virtual ~Gauss();
  virtual Module* clone(int deep);
  virtual void execute();

  void do_parallel(int proc);

  // event handlers for the window
  void DoMotion(int x, int y,int which);

  void DoDown(int x, int y, int button);
  void DoRelease(int x, int y, int button);

  virtual void Refresh();  
  void Resize();
  
  void tcl_command( TCLArgs&, void* );

  int makeCurrent(void);

};

extern "C" {
  Module* make_Gauss(const clString& id)
    {
      return scinew Gauss(id);
    }
}

//static clString module_name("Gauss");
//static clString widget_name("Gauss Widget");

Gauss::Gauss(const clString& id)
: Module("Gauss", id, Filter),
  sigma("sigma", id, this), size("size", id, this),
  hardware("hardware", id, this)
{
  // Create the input ports

  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  
  add_iport( inscalarfield);
  add_oport( outscalarfield);

  outgrid = new ScalarFieldRG;
  
  pbuf = 0;
  ctx = 0; // null for now - no window is bound yet
  bdown = -1;
  drawn = 0;  // glDrawpixels hasn't been called yet
  gen=99;

  
}

Gauss::Gauss(const Gauss& copy, int deep)
: Module(copy, deep),
  sigma("sigma", id, this),  size("size", id, this),
  hardware("hardware", id, this)
{
  NOT_FINISHED("Gauss::Gauss");
}

Gauss::~Gauss()
{
}

Module* Gauss::clone(int deep)
{
  return scinew Gauss(*this, deep);
}

void Gauss::do_parallel(int proc)
{
  int start = (outgrid->grid.dim1()-1)*proc/np;
  int end   = (proc+1)*(outgrid->grid.dim1()-1)/np;

  if (!start) start++;
  if (end == outgrid->grid.dim1()-1) end--;
    
  for(int x=start; x<end; x++)                 // 1 -> outgrid->grid.dim1()-1 
    for(int y=1; y<outgrid->grid.dim2()-1; y++){               // start -> end
      outgrid->grid(x,y,0) = (gauss[0]*ingrid->grid(x-1,y-1,0) + \
			      gauss[1]*ingrid->grid(x,y-1,0) + \
			      gauss[2]*ingrid->grid(x+1,y-1,0) + \
			      gauss[3]*ingrid->grid(x-1,y,0) + \
			      gauss[4]*ingrid->grid(x,y,0) + \
			      gauss[5]*ingrid->grid(x+1,y,0) + \
			      gauss[6]*ingrid->grid(x-1,y+1,0) + \
			      gauss[7]*ingrid->grid(x,y+1,0) + \
			      gauss[8]*ingrid->grid(x+1,y+1,0)); 
      }
}

static void do_parallel_stuff(void* obj,int proc)
{
  Gauss* img = (Gauss*) obj;

  img->do_parallel(proc);
}


void Gauss::execute()
{
 
  // get the scalar field and colormap...if you can
  ScalarFieldHandle sfieldh;
  if (!inscalarfield->get( sfieldh ))
    return;
  sfield=sfieldh.get_rep();

  ingrid = sfield->getRG();

  if (!ingrid  )   
    return;

  gen=ingrid->generation;
    
  if (gen!=outgrid->generation) {
    outgrid=new ScalarFieldRG(*ingrid);
    dgrid=new ScalarFieldRGfloat;
  } 
  
  if (ingrid) {
    width = ingrid->grid.dim2();
    height = ingrid->grid.dim1();
  } 
   
  
  outgrid->resize(height,width,1);
   
  dgrid = new ScalarFieldRGfloat;
  dgrid->resize(height,width,1);

  double maxval,minval;

/*  for (int x=0;x<width;x++)
    for (int y=0;y<height;y++) {
      if (ingrid->grid(y,x,0)>maxval)
	maxval=ingrid->grid(y,x,0);
    }
*/

  ingrid->compute_minmax();
  ingrid->get_minmax(minval,maxval);
  
  
  cerr << "max : " << maxval << "\n";
  int x;
  for (x=0;x<width;x++)
    for (int y=0;y<height;y++) {
      dgrid->grid(y,x,0) = float(ingrid->grid(y,x,0))/maxval; 
    }
    
  cerr << "Gauss - dims : " << width << " by " << height << "\n";

  drawn = 1;

  int usehardware = hardware.get(); 

  if (usehardware) {
    if (!makeCurrent() )
      return; 

    glXMakeCurrent(dpy,pbuf,ctx);
    
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
  }
  
  
  sig=sigma.get();
  siz=(unsigned char)size.get();

  cerr << "Creating gaussian filter : \n Sigma : " << sig << " Size : " << siz << "\n";
  
  int k = (siz-1)/2;
  gauss = new float[siz*siz];
  
  double sig2 = (sig)*(sig)*2;
  double sum=0;
  int i1;
  for (i1=-k;i1<=k;i1++)
    for (int j1=-k;j1<=k;j1++) {
      gauss[k+i1+(k+j1)*siz] = exp(-((i1*i1+j1*j1)/sig2));
      sum+=gauss[k+i1+(k+j1)*siz];
    }
  for (i1=0;i1<(siz*siz);i1++) {
    gauss[i1]=gauss[i1]/sum;
  }

  if (usehardware) {
  
    glConvolutionFilter2DEXT(GL_CONVOLUTION_2D_EXT,  \
			     GL_LUMINANCE, \
			     siz,siz, \
			     GL_LUMINANCE,GL_FLOAT,gauss); 
    glEnable(GL_CONVOLUTION_2D_EXT); 
    
    //  glPixelTransferf(GL_POST_CONVOLUTION_RED_BIAS_EXT,0.5);
    
    glDrawPixels(width,height,GL_LUMINANCE,GL_FLOAT,&dgrid->grid(0,0,0));
    
    glDisable(GL_CONVOLUTION_2D_EXT);
    
    //  glPixelTransferf(GL_RED_BIAS,-0.5);
    
    glReadPixels(0,0,width,height,GL_RED,GL_FLOAT,&dgrid->grid(0,0,0));
    
    //  glXSwapBuffers(dpy,win);

    for (x=0;x<width;x++)
      for (int y=0;y<height;y++)
	outgrid->grid(y,x,0)=(dgrid->grid(y,x,0)*maxval);

/*  int max = 0;
  int two32=2;
  for (x=0;x<29;x++)
    two32=two32*2;
  cerr << two32 << "\n";
  for (x=0;x<width;x++)
    for (int y=0;y<height;y++) {
      if (outgrid->grid(y,x,0)>max)
	max=outgrid->grid(y,x,0);
      outgrid->grid(y,x,0)=((outgrid->grid(y,x,0)-(two32)))*2;
    } 

  cerr << "two32 : " << two32 << "\n";
  cerr << "max : " << max << "\n"; */
  
    glXMakeCurrent(dpy,None,NULL);
    TCLTask::unlock();
  } else {
    np = Task::nprocessors();
    Task::multiprocess(np, do_parallel_stuff, this);
  }
  
  // Send out

  outscalarfield->send(outgrid);

}

void Gauss::Refresh()
{

}


void Gauss::Resize()
{

  // do a make current...

  if ( ! makeCurrent() )
    return;

  glXMakeCurrent(dpy,win,ctx);

  // make current...

  // Setup transformation for 1.0 -> width and height of the viewport
  
  glViewport(0,0,width,height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0,width,0,height,-1.0,1.0);
  glMatrixMode(GL_MODELVIEW);

  glEnable(GL_COLOR_MATERIAL);  // more state?
  glColorMaterial(GL_FRONT_AND_BACK,GL_DIFFUSE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);

  glViewport(0,0,width,height);
  glClear(GL_COLOR_BUFFER_BIT);

  glXMakeCurrent(dpy,None,NULL);


  TCLTask::unlock();
//  Refresh();
  
}

void Gauss::tcl_command( TCLArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for Gauss");
    return;
  }

  // mouse commands are motion, down, release

  if (args[1] == "resetraster") {
        
    Refresh();
  } 
  
  if (args[1] == "mouse") {
    int x,y,whichw,whichb;
    
    args[4].get_int(x);
    args[5].get_int(y);

    args[2].get_int(whichw); // which window it was in

    if (args[3] == "motion") {
      if (bdown == -1) // not buttons down!
	return;
      args[6].get_int(whichb); // which button it was
      DoMotion(x,y,whichb);
    } else {
      args[6].get_int(whichb); // which button it was
      if (args[3] == "down") {
	DoDown(x,y,whichb);
      } else { // must be release
	DoRelease(x,y,whichb);
      }
    }
  } else  if (args[1] == "resize") { // resize event!
    int whichwin;
    args[4].get_int(whichwin);
    args[2].get_int(winX);
    args[3].get_int(winY);
    Resize();  // kick of the resize function...
  } else if (args[1] == "expose") {
    int whichwin;
    args[2].get_int(whichwin);
    Resize(); // just sets up OGL stuff...
  }else {
    Module::tcl_command(args, userdata);
  }
}

void Gauss::DoMotion( int, int, int)
{

}

void Gauss::DoDown(int, int, int which)
{
  bdown = which;

}

void Gauss::DoRelease(int, int, int)
{
  bdown = -1;
}

int Gauss::makeCurrent(void)
{
  Tk_Window tkwin;

  // lock a mutex
  TCLTask::lock();


  if ((!pbuf) || (width>oldwidth) || (height>oldheight)) {
    cerr << "No pbuffer yet.. or creating a bigger one..\n";

    int attributes[9]={GLX_RENDER_TYPE_SGIX, GLX_RGBA_BIT_SGIX, GLX_RED_SIZE,\
		       4,GLX_GREEN_SIZE, 4, GLX_BLUE_SIZE, 4, None};


    GLXFBConfigSGIX *config1;
    int num;

    int pattr[3]={GLX_PRESERVED_CONTENTS_SGIX,True,None};
    clString myname(clString(".ui")+id);

    tkwin = Tk_NameToWindow(the_interp, myname(),Tk_MainWindow(the_interp));

    winX = Tk_Width(tkwin);
    winY = Tk_Height(tkwin);
    
    dpy = Tk_Display(tkwin);

    if (!dpy) {
      cerr << "Please hit Gauss' UI button before running (for the pbuffer)\n";
      TCLTask::unlock();
      return 0;
    }

    config1 = glXChooseFBConfigSGIX(dpy,Tk_ScreenNumber(tkwin),attributes,&num);
    cerr << "Got configs.." << num << "\n";
    pbuf = glXCreateGLXPbufferSGIX(dpy,*config1,width,height,pattr);

    ctx = glXCreateContextWithConfigSGIX(dpy,*config1,GLX_RGBA_TYPE_SGIX,NULL,1);

    oldwidth=width;
    oldheight=height;

    if(!ctx)
      {
	cerr << "Unable to create OpenGL Context!\n";
	TCLTask::unlock();
	return 0;
      }
    
    if (!pbuf) {
      cerr << "no pbuffer..\n";
      TCLTask::unlock();
      return 0;
    }
  } 

  return 1;
  
}

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.3  1999/08/25 03:48:55  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:59  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:52  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:31  dav
// Added image files to SCIRun
//
//

