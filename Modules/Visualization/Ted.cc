/*
 *  Ted.cc:  2D OpenGL Viewer
 *
 *  Written by:
 *    Scott Morris
 *    August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldRGshort.h>
#include <Datatypes/ScalarFieldRGfloat.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <TCL/TCLvar.h>
#include <TCL/TCLTask.h>

#include <GL/gl.h>
#include <GL/glu.h>

#include <Geom/GeomOpenGL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

// tcl interpreter corresponding to this module

extern Tcl_Interp* the_interp;

// the OpenGL context structure

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);


class Ted : public Module {
  ScalarFieldIPort *inscalarfield;
  ScalarFieldIPort *inscalarfield2;   // Input for mask
  ScalarFieldIPort *inscalarfield3;   // Input for mask2
  ScalarFieldOPort *outscalarfield;   // Output the masked image
  
  int grid_id;

  ScalarFieldRG *mask,*mask2; // this only works on regular grids for chaining
  ScalarFieldRGfloat* backup;
  ScalarFieldRG* ingrid32;
  ScalarFieldRG* ingrid32b;
  ScalarFieldRG* outmask;
  ScalarFieldRGfloat *ingridf;
    
  ScalarField* sfield;  // The three input scalarfields
  ScalarField* sfield2;
  ScalarField* sfield3;

  ScalarFieldHandle handle;

  GLXContext		ctx;    // OpenGL Contexts
  Display		*dpy;
  Window		win;

  int                   winX,winY; // size of window in pixels
  int                   bdown;
  int                   width,height,zdim;  // dims of image
  int                   drawn,rgb,m1,m2;  

  int np;
  
  TCLdouble zoom;
  TCLdouble normal;
  TCLdouble negative;
  
  double zoomval;             // Pixelzoom value
  double px,py;               // Raster Position, used for Pan
  int oldx,oldy,oldpx,oldpy;  // old values for Panning
  double maxval;              // Max image value, used for scaling
  
public: 
  Ted(const clString& id);
  Ted(const Ted&, int deep);
  virtual ~Ted();
  virtual Module* clone(int deep);

  void abs_parallel(int proc);   // Takes care of negatives, finds max
  void scale_parallel(int proc); // scales, or normalizes
  void mask_parallel(int proc);  // Creates the mask(s)
  
  virtual void execute();

  // event handlers for the window
  void DoMotion(int x, int y,int which);

  void output_grid(int x,int y);   // text output of a square of grid #'s
  void DoDown(int x, int y, int button);
  void DoRelease(int x, int y, int button);

  virtual void Refresh();   // my_draw()
  void Resize();
  
  void tcl_command( TCLArgs&, void* );

  int makeCurrent(void);


};

extern "C" {
  Module* make_Ted(const clString& id)
    {
      return scinew Ted(id);
    }
};

static clString module_name("Ted");
static clString widget_name("Ted Widget");

Ted::Ted(const clString& id)
: Module("Ted", id, Filter),
  zoom("zoom", id, this),
  normal("normal", id, this),
  negative("negative", id, this)
{
  // Create the input ports

  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  inscalarfield2 = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  inscalarfield3 = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  
  add_iport( inscalarfield);
  add_iport( inscalarfield2);
  add_iport( inscalarfield3);
  add_oport( outscalarfield);
  
  ctx = 0;    // null for now - no window is bound yet
  bdown = -1;
  drawn = 0;  // glDrawpixels hasn't been called yet
  m1=m2=0;    // No masks.. yet
  zoomval=1;  
  px=0;
  py=0;       // no pan and no zoom
  oldx=0;
  oldy=0;

  ingridf = new ScalarFieldRGfloat;

}

Ted::Ted(const Ted& copy, int deep)
: Module(copy, deep),
  zoom("zoom", id, this),
  normal("normal", id, this),
  negative("negative", id, this)
{
  NOT_FINISHED("Ted::Ted");
}

Ted::~Ted()
{
}

Module* Ted::clone(int deep)
{
  return scinew Ted(*this, deep);
}

// Make these three do 3 channels (RGB)


void Ted::abs_parallel(int proc)  // Scales up negatives 
{
  int start = (width-1)*proc/np;
  int end   = (proc+1)*(width-1)/np;

  for (int z=0; z<zdim; z++)
    for (int x=start; x<end; x++)
      for (int y=0; y<height; y++) {
	if ((ingrid32->grid(y,x,z)<0) && (negative.get()))
	  ingrid32b->grid(y,x,z) = ingrid32->grid(y,x,z)- \
	    2*ingrid32->grid(y,x,z);
	else ingrid32b->grid(y,x,z)=ingrid32->grid(y,x,z);
	if (normal.get())
	  if (ingrid32b->grid(y,x,z)>maxval)
	    maxval=ingrid32b->grid(y,x,z);
      }
}

void Ted::scale_parallel(int proc)  // Scales image to 1.0 
{
  int start = (width-1)*proc/np;
  int end   = (proc+1)*(width-1)/np;

  for (int z=0; z<zdim; z++)
    for (int x=start; x<end; x++)
      for (int y=0; y<height; y++) {
	if (ingrid32->grid(y,x,z)>=0)
	  ingridf->grid(y,x,z)=float(ingrid32b->grid(y,x,z))/maxval;
	else
	  if (negative.get())
	    ingridf->grid(y,x,z)=1.0 - \
	      float(ingrid32b->grid(y,x,z))/maxval;
	  else
	  ingridf->grid(y,x,z)=0;
      }
}

void Ted::mask_parallel(int proc)  // creates the mask 
{
  int start = (width-1)*proc/np;
  int end   = (proc+1)*(width-1)/np;

  for (int x=start; x<end; x++)
    for (int y=0; y<height; y++) {
      backup->grid(y,x,2)=backup->grid(y,x,1)=backup->grid(y,x,0)=\
	ingridf->grid(y,x,0);
      if (mask->grid(y,x,0)!=0) { 
	backup->grid(y,x,0)=1.0;
	backup->grid(y,x,1)=0;
	backup->grid(y,x,2)=0;
      }
      if (m2)
	if (mask2->grid(y,x,0)!=0) { 
	  if (!(mask->grid(y,x,0)!=0))
	    backup->grid(y,x,0)=0;
	  backup->grid(y,x,1)=0;
	  backup->grid(y,x,2)=1.0;
	}
      outmask->grid(y,x,0)=backup->grid(y,x,0);
      outmask->grid(y,x,1)=backup->grid(y,x,1);
      outmask->grid(y,x,2)=backup->grid(y,x,2);
    }
}

static void abs_starter(void* obj,int proc)
{
  Ted* img = (Ted*) obj;

  img->abs_parallel (proc);
}

static void scale_starter(void* obj,int proc)
{
  Ted* img = (Ted*) obj;

  img->scale_parallel (proc);
}

static void mask_starter(void* obj,int proc)
{
  Ted* img = (Ted*) obj;

  img->mask_parallel (proc);
}

void Ted::execute()
{

  // Get the main scalarfield..

  ScalarFieldHandle sfieldh;
  if (!inscalarfield->get( sfieldh ))
    return;
  sfield=sfieldh.get_rep();

  ingrid32=sfield->getRG();
  
  if (!ingrid32)
    return;

  maxval=0;

  np = Task::nprocessors();

  ingrid32b = new ScalarFieldRG(*ingrid32);
  width = ingrid32->grid.dim2();
  height = ingrid32->grid.dim1();
  zdim = ingrid32->grid.dim3();
  
  cerr << "TED - dims : " << width << " by " << height << "\n";

  rgb = (ingrid32->grid.dim3()==3);
  
/*  if (!normal.get()) {   // compute max value for 32bit signed int
    maxval=2;
    for (int temp=0;temp<30;temp++)
      maxval = maxval*2;
    maxval=maxval-1;
  }
*/

  if (!normal.get()) {
    maxval=255;  // Assume 8 bit images by default
  }
  
  Task::multiprocess(np, abs_starter, this);

  ingridf = new ScalarFieldRGfloat;
  ingridf->resize(height,width,zdim);
  ingridf->compute_bounds();
  Point pmin(0,0,0),pmax(height,width,zdim);
  ingridf->set_bounds(pmin,pmax);
  cerr << "Maximum value : " << maxval << "\n";
  
  Task::multiprocess(np, scale_starter, this);

  // Check for any overlays

  ScalarFieldHandle sfieldh2;
  m1=m2=0;
  if (inscalarfield2->get( sfieldh2 )) {
    cerr << "Hey.. an overlay!\n";
    sfield2=sfieldh2.get_rep();

    mask = sfield2->getRG();
    if (mask) {
	m1 = ((mask->grid.dim2()==width) && (mask->grid.dim1()==height));
	if (!m1)
	  cerr << "First mask is wrong size..\n";
    }

    ScalarFieldHandle sfieldh3;   // Check for second overlay
    if (inscalarfield3->get( sfieldh3 )) {
      cerr << "Hey.. a second overlay!\n";
      sfield3=sfieldh3.get_rep();

      mask2 = sfield3->getRG();
      if (mask2) {
	m2 = ((mask2->grid.dim2()==width) && (mask2->grid.dim1()==height));
	if (!m2)
	  cerr << "Second mask is wrong size..\n";
      }
    }
  }
  if (m1) {
    backup=new ScalarFieldRGfloat(*ingridf);
    backup->resize(ingrid32->grid.dim1(),ingrid32->grid.dim2(),3);

    outmask=new ScalarFieldRG(*ingrid32);
    outmask->resize(ingrid32->grid.dim1(),ingrid32->grid.dim2(),3);

    Task::multiprocess(np, mask_starter, this);

    handle = outmask;
  } else {

    handle = ingrid32;
    mask = 0;
    cerr << "No Mask(s)\n";
  }
    
  drawn = 1;

  Refresh();

  // Send out the overlayed image, or just the image that came in

  outscalarfield->send(handle);

}

void Ted::Refresh()
{
  if (!makeCurrent() )
    return;

  glXMakeCurrent(dpy,win,ctx);

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

//  zoomval=zoom.get();
  
  glPixelZoom(zoomval,zoomval); 

  /* Nate Robbin's glBitmap trick to get it to display outside the window */
  glRasterPos2i(0,0);
  glBitmap(0, 0, 0, 0, px, py, NULL);
  
  if (drawn) {

    if ((!m1) && (!rgb)) {
      glDrawPixels(width,height,GL_LUMINANCE,GL_FLOAT,&ingridf->grid(0,0,0));
    } else
    if (rgb) {
      glDrawPixels(width,height,GL_RGB,GL_FLOAT,&ingridf->grid(0,0,0));
      cerr << "Drawing color RGB Image...\n";
    }
    if (m1) {
      cerr << "Drawing with overlay...\n";
      glDrawPixels(width,height,GL_RGB,GL_FLOAT,&backup->grid(0,0,0));
    }
    

  glXSwapBuffers(dpy,win);

  }
  glXMakeCurrent(dpy,None,NULL);
  TCLTask::unlock();
}


void Ted::Resize()
{

  // do a make current...

  if ( ! makeCurrent() )
    return;

  glXMakeCurrent(dpy,win,ctx);

  // make current...

  // Setup transformation for 1.0 -> width and height of the viewport
  
  glViewport(0,0,winX,winY);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrtho(0,winX,0,winY,-1.0,1.0);
  glMatrixMode(GL_MODELVIEW);

  glEnable(GL_COLOR_MATERIAL);  // more state?
  glColorMaterial(GL_FRONT_AND_BACK,GL_DIFFUSE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);

  glViewport(0,0,winX,winY);
  glClear(GL_COLOR_BUFFER_BIT);

  glXMakeCurrent(dpy,None,NULL);


  TCLTask::unlock();
  Refresh();
  
}

void Ted::tcl_command( TCLArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for Ted");
    return;
  }

  // mouse commands are motion, down, release

  if (args[1] == "viewgrid") {
    int x,y;
    args[2].get_int(x);
    args[3].get_int(y);

    output_grid(x,y);
  } else
  
  if (args[1] == "resetraster") {
    px=py=0;
    
    Refresh();
  } else

  if (args[1] == "justrefresh") {
    args[2].get_double(zoomval);
    Refresh();
  } else
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

void Ted::DoMotion( int x, int y,int which)
{
  y = winY - y;

  if (which==2) {
    px = oldpx + (x-oldx);
    py = oldpy + (y-oldy);
//    cerr << px << " " << py << "\n";
  }
  
  Refresh();

}

void Ted::output_grid(int x,int y)
{
  if ((ingrid32) && (x < width) && (y < height) && (x>=0) && (y>=0))
    cerr << x << ":" << y << " value : " << ingrid32->grid(y,x,0) << "\n";

  if ((ingrid32) && (x < width-1) && (y<height-1) && (x>0) && (y>0)) {
    cerr << "3x3 Grid : " <<   ingrid32->grid(y+1,x-1,0) << " " \
	 <<  ingrid32->grid(y+1,x,0) << " "\
         <<  ingrid32->grid(y+1,x+1,0) << "\n";
    cerr << "           " <<  ingrid32->grid(y,x-1,0) << " " \
	 <<  ingrid32->grid(y,x,0) << " "\
         <<  ingrid32->grid(y,x+1,0) << "\n";
    cerr << "           " <<  ingrid32->grid(y-1,x-1,0) << " " \
	 <<  ingrid32->grid(y-1,x,0) << " "\
         <<  ingrid32->grid(y-1,x+1,0) << "\n";
  }
}

void Ted::DoDown(int x, int y, int which)
{
  y = winY - y;

  switch (which) {
  case 1:

   cerr << "Clicked at : " << x/zoomval-px/zoomval << " " <<
    y/zoomval-py/zoomval << ";\n";
   
   x = (x/zoomval)-(px/zoomval);
   y = (y/zoomval)-(py/zoomval);

   output_grid(x,y);
   
   break;
  case 2:
    cerr << "Starting Pan..\n";

    oldpx=px;
    oldpy=py;
    oldx=x;
    oldy=y;
    break;
  }
    
  
  bdown = which;
}

void Ted::DoRelease(int x, int y, int)
{
  bdown = -1;
}

int Ted::makeCurrent(void)
{
  Tk_Window tkwin;

  // lock a mutex
  TCLTask::lock();

  if (!ctx) {
    cerr << "Context is not defined!\n";
    clString myname(clString(".ui")+id+".f.gl1.gl");
    tkwin = Tk_NameToWindow(the_interp, myname(),Tk_MainWindow(the_interp));

    if (!tkwin) {
      cerr << "Unable to locate window!\n";
      
      // unlock mutex
      TCLTask::unlock();
      return 0;
    }
    winX = Tk_Width(tkwin);
    winY = Tk_Height(tkwin);

    dpy = Tk_Display(tkwin);
    win = Tk_WindowId(tkwin);

    ctx = OpenGLGetContext(the_interp,myname());

    // check if it was created
    if(!ctx)
      {
	cerr << "Unable to create OpenGL Context!\n";
	TCLTask::unlock();
	return 0;
      }
  }	

  // ok, 3 contexts are created - now you only need to
  // do the other crap...

  return 1;
  
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<GeomMaterial*>;
template class Array1<double>;

#endif







