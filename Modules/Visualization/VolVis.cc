/*
 *  VolVis.cc:  Volume visualization module on structured grids.
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */


#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>


#include <Geom/View.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>



#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <TCL/TCLTask.h>
#include <TCL/TCL.h>

#include <Datatypes/ColormapPort.h>


#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <iostream.h>
#include <string.h>
#include <stdlib.h>


#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Geom/GeomOpenGL.h>

#include <Geom/TCLView.h>

#include <Modules/Visualization/LevoyVis.h>
#include "kuswik.h"

#define TRUE 1
#define FALSE 0

const View homeview(Point(2, 0.5, 0.5), Point(0.5, 0.5, 0.5), Vector(.0,.0,1), 45);

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);


/**************************************************************
 *
 *
 *
 **************************************************************/


class VolVis : public Module {

    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    ScalarFieldHandle last_homeSFHandle;	// last input fld
    ScalarFieldRG* last_homeSFRGrid;		// just a convenience

    double x_pixel_size;
    double y_pixel_size;
    double z_pixel_size;
    int x_win_min;
    int y_win_min;
    int drawing;

    // the image is stored in this 2d-array.  it contains
    // pixel values arranged in the order row then column
    Array2<char> image;

    Array2<CharColor> Image;

    clString myid;
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    int tcl_execute;
public:
  /*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

    // holds the value of current viewing vector
    View myview;

    // the size in pixels of output window
    
    int rasterX, rasterY;

    // newly added
    TCLint interactiveRasterX, interactiveRasterY;
    TCLint curX, curY;
    TCLint minSV, maxSV;
    TCLdouble xpos, ypos;

    TCLdouble node1x, node1y;
    TCLdouble node2x, node2y;
    TCLdouble node3x, node3y;
    TCLdouble node4x, node4y;
    TCLdouble node5x, node5y;
    TCLdouble t1x, t1y;

    double Xvals[5];
    double Yvals[5];
    
    Vector rayIncrementU, rayIncrementV;
//    double rayStep;
    Point eye;
    double farthest;

    // holds the values supplied by the tcl interface
    TCLView interactiveView;

    // holds the values of the last interactive view
    // (so that no execution takes place if the view didn't
    // change)
    View lastInteractiveView;
  
    ColormapIPort* colormapport;

    Color backgroundColor;

    TCLint interactiveRed;
    TCLint interactiveGreen;
    TCLint interactiveBlue;

    int cmapGeneration;

  /*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
  
    VolVis(const clString& id);
    VolVis(const VolVis&, int deep);
    virtual ~VolVis();
    virtual Module* clone(int deep);
    virtual void execute();
    void set_str_vars();
    void tcl_command( TCLArgs&, void * );
    void redraw_all();
    int makeCurrent();

    int ValidHandle ( ScalarFieldHandle& homeSFHandle,
			 ScalarFieldRG **homeSFRGrid,
		     ColormapHandle& cmap, int& cmapFlag );
  };


static VolVis* current_drawer=0;

extern "C" {
Module* make_VolVis(const clString& id)
{
    return scinew VolVis(id);
}
};

VolVis::VolVis(const clString& id)
: Module("VolVis", id, Source), tcl_execute(0),
  interactiveView("interactiveView", id, this),
  interactiveRasterX("rasterX", id, this),
  interactiveRasterY("rasterY", id, this),
  curX("curX", id, this),
  curY("curY", id, this),
  minSV("minSV", id, this),
  maxSV("maxSV", id, this),
  xpos("xpos", id, this),
  ypos("ypos", id, this),
  node1x("n1x", id, this),
  node2x("n2x", id, this),
  node3x("n3x", id, this),
  node4x("n4x", id, this),
  node5x("n5x", id, this),
  node1y("n1y", id, this),
  node2y("n2y", id, this),
  node3y("n3y", id, this),
  node4y("n4y", id, this),
  node5y("n5y", id, this),
  t1x("t1x", id, this),
  t1y("t1y", id, this),
  interactiveRed("red", id, this),
  interactiveGreen("green", id, this),
  interactiveBlue("blue", id, this),
  drawing(0)
{
  // Create the input port
    colormapport=scinew ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
    add_iport(colormapport);

    // Create the input port
    myid=id;
    iport = scinew ScalarFieldIPort(this, "HOMESFRGRID", ScalarFieldIPort::Atomic);
    add_iport(iport);

//    rasterX = 64;
//    rasterY = 128;

    rasterX = rasterY = 100;
    interactiveRasterX.set( rasterX );
    interactiveRasterY.set( rasterY );

    myview = homeview;
    lastInteractiveView = homeview;
    interactiveView.set( myview );

    Color temp( 0, 0, 0 );
    backgroundColor = temp;

    curX.set( 0 );
    curY.set( 0 );

    minSV.set( 0 );
    maxSV.set( 0 );

    Xvals[0] = 0;
    Yvals[0] = 202;

    Xvals[1] = 40;
    Yvals[1] = 202;
    
    Xvals[2] = 55;
    Yvals[2] = 150;

    Xvals[3] = 70;
    Yvals[3] = 202;

    Xvals[4] = 202;
    Yvals[4] = 202;

    node1x.set(0);
    node1y.set(202);

    node2x.set(40);
    node2y.set(202);

    node3x.set(55);
    node3y.set(150);

    node4x.set(70);
    node4y.set(202);

    node5x.set(202);
    node5y.set(202);
  }

VolVis::VolVis(const VolVis& copy, int deep)
: Module(copy, deep), tcl_execute(0), drawing(0),
  interactiveView("interactiveView", id, this),
  interactiveRasterX("rasterX", id, this),
  interactiveRasterY("rasterY", id, this),
  curX("curX", id, this),
  curY("curY", id, this),
  maxSV("maxSV", id, this),
  minSV("minSV", id, this),
  xpos("xpos", id, this),
  ypos("ypos", id, this),
  node1x("n1x", id, this),
  node2x("n2x", id, this),
  node3x("n3x", id, this),
  node4x("n4x", id, this),
  node5x("n5x", id, this),
  node1y("n1y", id, this),
  node2y("n2y", id, this),
  node3y("n3y", id, this),
  node4y("n4y", id, this),
  node5y("n5y", id, this),
  t1x("t1x", id, this),
  t1y("t1y", id, this),
  interactiveRed("red", id, this),
  interactiveGreen("green", id, this),
  interactiveBlue("blue", id, this)
{
    NOT_FINISHED("VolVis::VolVis");
}

VolVis::~VolVis()
{
}

Module*
VolVis::clone(int deep)
{
    return scinew VolVis(*this, deep);
}

int
VolVis::ValidHandle ( ScalarFieldHandle &homeSFHandle,
			 ScalarFieldRG **homeSFRGrid, ColormapHandle &cmap,
			int& cmapflag)
{
  // get the scalar field handle
  iport->get(homeSFHandle);
  
  // Make sure this is a valid scalar field

  if ( ! homeSFHandle.get_rep() )
    return 0;

  if ( lastInteractiveView == interactiveView.get() )
    cout << "the view hasn't changed\n";
  else
    cout << "view is changed!!!\n";


  Color c( interactiveRed.get()/255.0, interactiveGreen.get()/255.0,
	  interactiveBlue.get()/255.0 );

  // if tcl did not make it execute then why execute?
  // if we are using the same scalar field like last time, why execute?

  // newly added
  // check if the interactive raster size has changed

  if ( ( !tcl_execute ) &&
      ( homeSFHandle.get_rep() == last_homeSFHandle.get_rep() ) &&
      ( lastInteractiveView == interactiveView.get() )          &&
      ( rasterX == interactiveRasterX.get() )                   &&
      ( rasterY == interactiveRasterY.get() )                   &&
      ( backgroundColor == c                ) )
    return 0;
//      ( cmap->generation == cmapGeneration  ) )
  
//  cmapGeneration = cmap->generation;

  // make sure scalar field is a regular grid
  if ( ( (*homeSFRGrid) = homeSFHandle->getRG()) == 0)
    return 0;

  // i've got to assign min and max Scalar values somewhere....
  // i'll do it here since i got the sfhandle

  int max=-9999;
  int min=9999;
	for (int i=0; i<(*homeSFRGrid)->nx; i++)
	  for (int j=0; j<(*homeSFRGrid)->ny; j++)
	    for (int k=0; k<(*homeSFRGrid)->nz; k++) {
	      char c=(char)(*homeSFRGrid)->grid(i,j,k);
	      if (c>max)max=c;
	      if (c<min)min=c;
//      fout << c;
	    }
//	cout << "Max value from Scalar field was: " << max << "\n";
//	cout << "Min value from Scalar field was: " << min << "\n";

  minSV.set( min );
  maxSV.set( max );

  // must have a colormap in order to do anything
  if ( colormapport->get(cmap) )
    cmapflag = 1;

    return 1;
}

void
VolVis::execute()
{
  ScalarFieldHandle homeSFHandle;
  ScalarFieldRG *homeSFRGrid;
  ColormapHandle cmap;

  int cmapflag = 0;
  // make sure that the input port is valid

  // make SURE TO RESET VARS FROM TCL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  reset_vars();

  if ( ! ValidHandle ( homeSFHandle, &homeSFRGrid, cmap, cmapflag ) )
    return;

  lastInteractiveView = interactiveView.get();
  myview = lastInteractiveView;

  // newly added

  rasterX = interactiveRasterX.get();
  rasterY = interactiveRasterY.get();

  at();
  cout << "rasters: " << rasterX << " " << rasterY << endl;

  // newly added

  Color ccc( interactiveRed.get()/255.0, interactiveGreen.get()/255.0,
	    interactiveBlue.get()/255.0 );
  backgroundColor = ccc;
	 
  // initialize the image array
  
  image.newsize( rasterY, rasterX );
  image.initialize(0);

  CharColor fff;

  Image.newsize( rasterY, rasterX );
  Image.initialize(fff);
  
  // do the work => trace each ray

  Levoy levoyModule ( homeSFRGrid );
  levoyModule.AssignRaster ( rasterX, rasterY );

  cout << "this is the cmapflag" << cmapflag << endl;
  levoyModule.colormapFlag = cmapflag;
  levoyModule.cmap = cmap;

  levoyModule.minSV = minSV.get();
  levoyModule.maxSV = maxSV.get();
  
  cout << "before trace rays\n";

  at();

  cout << "dim1 : " << Image.dim1();
  cout << "\n dim2 : " << Image.dim2() << endl;
  
  levoyModule.TraceRays ( myview, image, Image, backgroundColor,
			 Xvals, Yvals );

  cout << "after trace rays\n";

  cout << "dim1 : " << Image.dim1();
  cout << "\n dim2 : " << Image.dim2() << endl;
  
  // memorize the last scalar field => will not execute
  // with no purpose
  
  last_homeSFHandle=homeSFHandle;
  last_homeSFRGrid=homeSFRGrid;
    
    Point pmin;
    Point pmax;

    x_pixel_size = 1;
    y_pixel_size = 1;
    z_pixel_size = 1;

    x_win_min = 0;
    y_win_min = rasterY;

  cout << "before redraw all\n";
  at();
    redraw_all();
  cout << "after redreaw sakjdl\n";
  at();

    // tcl has executed
    tcl_execute=0;
}

/*
 * redraws but does not recalculate everything
 */

void
VolVis::redraw_all()
{
  if (!makeCurrent()) return;

//  glClearColor(0, 1, 0, 0);
  glClearColor( backgroundColor.r(), backgroundColor.g(), backgroundColor.b(), 0 );
  
  glClear(GL_COLOR_BUFFER_BIT);
  
  // if the handle was empty, just flush the buffer (to clear the window)
    // and return.
  
  if (!last_homeSFHandle.get_rep())
    {
      glFlush();
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }
  
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  glPixelZoom(x_pixel_size, y_pixel_size);
  
  glRasterPos2i( x_pixel_size, y_pixel_size * (rasterY+1) );
  
  int loop;


      Color c(1, 0, 1);

#if 0  
  for ( loop = 0; loop < rasterX/2; loop++ )
    {
      image(loop, 10) = (char)50;
      image(loop, 50) = (char)50;

    }
#endif  

  
//  char *pixels=&(image(0,0));

  void *joy = &(Image(0,0));
  char *pixels = (char*) joy;

  // make sure that the routine to draw pixels is given the raster y first.
  // it wants width, then height
  
//  glDrawPixels( rasterX, rasterY, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels );
  glDrawPixels( rasterX, rasterY, GL_RGB, GL_UNSIGNED_BYTE, pixels );
  
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
}	    

void
VolVis::tcl_command(TCLArgs& args, void* userdata) {

  if (args[1] == "redraw_all" && !drawing)
      {
	drawing=1;
	reset_vars();
	redraw_all();
	drawing=0;
      }
    else if ( args[1] == "hope" )
      {
	reset_vars();
	cout <<"hello says hope " <<  t1x.get() << endl;
	cout <<"hello says hope " <<  t1y.get() << endl;
      }
  
    else if ( args[1] == "wanna_exec" )
      {
	reset_vars();
	tcl_execute = 1;
	want_to_execute();
      }
    else if ( args[1] == "incredible" )
      {
	reset_vars();
	int x, y, width;
	
	// the empty parentheses after the args[1-3] are used
	// to convert from clString to char*

	if ( !args[2].get_int(x) )
	  {
	    args.error("Error parsing X");
	    return;
	  }
	
	if ( !args[3].get_int(y) )
	  {
	    args.error("Error parsing Y");
	    return;
	  }
	
	if ( !args[4].get_int(width) )
	  {
	    args.error("Error parsing width");
	    return;
	  }
	
	xpos.set( 1.0 * x * (maxSV.get() - minSV.get()) / width * 1.0 );
	ypos.set( 1.0 * (width - y) / width );
      }
    else if ( args[1] == "get_data" )
      {
      reset_vars();

      cout << node1x.get() << endl;
      cout << node1y.get() << endl;

      cout << node2x.get() << endl;
      cout << node2y.get() << endl;
      
      cout << node3x.get() << endl;
      cout << node3y.get() << endl;
      
      cout << node4x.get() << endl;
      cout << node4y.get() << endl;

      Xvals[0] = node1x.get();
      Xvals[1] = node2x.get();
      Xvals[2] = node3x.get();
      Xvals[3] = node4x.get();
      Xvals[4] = node5x.get();

      Yvals[0] = node1y.get();
      Yvals[1] = node2y.get();
      Yvals[2] = node3y.get();
      Yvals[3] = node4y.get();
      Yvals[4] = node5y.get();
    }
    else
    {
      Module::tcl_command(args, userdata);
    }
}

/* this procedure initializes an open gl window. */

int
VolVis::makeCurrent() {

  // lock a mutex
  TCLTask::lock();

  // associate this with the "USER INTERFACE" button
  clString myname(clString(".ui")+id+".gl.gl");

  // find the tk window token for the window

  tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));

//  cout << "my name returned " << myname() << endl;

  // check if a token was associated
  if(!tkwin)
    {
      cerr << "Unable to locate window!\n";

      // unlock the mutex
      TCLTask::unlock();
      return 0;
    }
  
  // X-display for window
  dpy=Tk_Display(tkwin);

  // get the X-id for window
  win=Tk_WindowId(tkwin);

  // create an open gl context
  cx=OpenGLGetContext(the_interp, myname());

  // check if it was created
  if(!cx)
    {
      cerr << "Unable to create OpenGL Context!\n";
      TCLTask::unlock();
      return 0;
    }

  current_drawer=this;

  // sets up a bunch of stuff for OpenGL
  
  if ( ! glXMakeCurrent(dpy, win, cx) )
    cerr << "*glXMakeCurrent failed.\n";

  // Clear the screen...
  
  glViewport(0, 0, 599, 599);
  
  glMatrixMode(GL_PROJECTION);
  
  glLoadIdentity();
  
  glOrtho(0, 599, 599, 0, -1, 1);
  
  glMatrixMode(GL_MODELVIEW);
  
  glLoadIdentity();
  
  return 1;
}
