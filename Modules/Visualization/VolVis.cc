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


#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColormapPort.h>


#include <Geom/View.h>
#include <Geom/TCLView.h>
#include <Geom/TCLGeom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <Geom/GeomOpenGL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>


#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCL.h>

#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <iostream.h>
#include <stdlib.h>

#include <Modules/Visualization/LevoyVis.h>
#include "kuswik.h"

// the initial view data

const View homeview
(Point(2, 0.5, 0.5), Point(0.5, 0.5, 0.5), Vector(.0,.0,1), 45);

// the_interp is the tcl interpreter corresponding to this
// module

extern Tcl_Interp* the_interp;

// the OpenGL context structure

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);


/**************************************************************
 *
 * the Volume Visualization class.  provides for the interface
 * with TCL and Levoy volume visualization code, as well as
 * displays the image of a volume.
 *
 **************************************************************/


class VolVis : public Module {

  // scalar field input port, provides the 3D image data
  
  ScalarFieldIPort *iport;

  // color map input port, provides scalar field to color map
  // if non-existant, white is the color used

  ColormapIPort* colormapport;

  // handle to the scalar field
  
  ScalarFieldHandle homeSFHandle;

  // remember last generation of scalar field

  int homeSFgeneration;
  
  // OpenGL window size
  
  int ViewPort;

  // the size in pixels of output window
  
  double x_pixel_size;
  double y_pixel_size;

  

  // the view: eyepoint, atpoint, up vector, and field of view
  
  TCLView iView;

  // background color

  TCLColor ibgColor;

  // raster dimensions
  
  TCLint iRasterX, iRasterY;

  // min and max scalar values

  TCLint minSV, maxSV;

  // a list of {x,y}positions {x=scalar value, y=opacity}
  // associated with the nodes of "Opacity map" widget
  
  TCLstring Xarray, Yarray;

  // projection type

  TCLint projection;

  
  // raster dimensions

  int rasterX, rasterY;

  // background color
  
  Color bgColor;

  // arrays with scalar value -- opacity mapping
  
  Array1<double> Opacity;
  Array1<double> ScalarVal;
  
  double    Xvalues[CANVAS_WIDTH], Yvalues[CANVAS_WIDTH];

  // the number of nodes
  
  int       NodeCount;
  
  // the image is stored in this 2d-array.  it contains
  // pixel values arranged in the order row then column

  Array2<CharColor> Image;



  // TEMP?  what are these used for anyways?

  Display* dpy;


  
  // mutex to prevent problems related to changes of the
  // Image array

  Mutex imagelock;

  
  
  // initialize an OpenGL window
  
  int makeCurrent();

  int Validate ( ScalarFieldRG **homeSFRGrid );

  // place the {x,y}positions supplied by tcl-lists
  // into the arrays.

  void UpdateTransferFncArray( clString x, clString y );

public:

  // constructor
  
  VolVis(const clString& id);

  // copy constructor
  
  VolVis(const VolVis&, int deep);

  // destructor
  
  virtual ~VolVis();

  // clones the VolVis module
  
  virtual Module* clone(int deep);

  // calculate and display the new image
  
  virtual void execute();

  // process commands sent from tcl
  
  void tcl_command( TCLArgs&, void * );

  // redraw the OpenGL window
  
  void redraw_all();
  
};




/**************************************************************
 *
 * TEMP!
 *
 **************************************************************/

static VolVis* current_drawer=0;



/**************************************************************
 *
 *
 *
 **************************************************************/

extern "C" {
  Module* make_VolVis(const clString& id)
    {
      return scinew VolVis(id);
    }
};




/**************************************************************
 *
 * constructor
 *
 **************************************************************/

VolVis::VolVis(const clString& id)
: Module("VolVis", id, Source),
  iView("View", id, this),
  iRasterX("rasterX", id, this),
  iRasterY("rasterY", id, this),
  minSV("minSV", id, this),
  maxSV("maxSV", id, this),
  ibgColor("bgColor", id, this),
  projection("project", id, this),
  Xarray("Xarray", id, this),
  Yarray("Yarray", id, this)
{
  // Create a Colormap input port
  
  colormapport=scinew ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
  add_iport(colormapport);

  // Create a ScalarField input port
  
  iport = scinew ScalarFieldIPort(this, "RGScalarField", ScalarFieldIPort::Atomic);
  add_iport(iport);

  // initialize raster size
  
  iRasterX.set( 100 );
  iRasterY.set( 100 );

  iView.set( homeview );

  // initialize OpenGL window size (viewport size)
  
  ViewPort = 600;

  x_pixel_size = 1;
  y_pixel_size = 1;

  Xarray.set("0 40 55 70 200");
  Yarray.set("200 200 150 200 200");

  Color temp(0.,0.,0.);
  ibgColor.set( temp );
  bgColor = temp;

  homeSFgeneration = -1;

  minSV.set(0);
  maxSV.set(121);

  rasterX = 100;
  rasterY = 100;

  Image.newsize(100,100);
  Image.initialize( bgColor );
}



/**************************************************************
 *
 * copy constructor
 *
 **************************************************************/

VolVis::VolVis(const VolVis& copy, int deep)
: Module(copy, deep),
  iView("View", id, this),
  iRasterX("rasterX", id, this),
  iRasterY("rasterY", id, this),
  maxSV("maxSV", id, this),
  minSV("minSV", id, this),
  ibgColor("bgColor", id, this),
  projection("project", id, this),
  Xarray("Xarray", id, this),
  Yarray("Yarray", id, this)
{
  NOT_FINISHED("VolVis::VolVis");
}



/**************************************************************
 *
 * destructor
 *
 **************************************************************/

VolVis::~VolVis()
{
}



/**************************************************************
 *
 * clones this class
 *
 **************************************************************/

Module*
VolVis::clone(int deep)
{
  return scinew VolVis(*this, deep);
}



/**************************************************************
 *
 * make sure that the scalar field handle points to valid
 * data.  initialize some variables.
 *
 **************************************************************/

int
VolVis::Validate ( ScalarFieldRG **homeSFRGrid )
{
  double min, max;
  
  // make sure TCL variables are updated
  
  reset_vars();

  // get the scalar field handle
  
  iport->get(homeSFHandle);

  // Make sure this is a valid scalar field

  if ( ! homeSFHandle.get_rep() )
    return 0;

  // make sure scalar field is a regular grid
  
  if ( ( (*homeSFRGrid) = homeSFHandle->getRG()) == 0)
    return 0;

  if ( homeSFgeneration != homeSFHandle->generation )
  {
    // remember the generation
  
    homeSFgeneration = homeSFHandle->generation;

    // get the min, max values of the new field

    homeSFHandle->get_minmax( min, max );

    // set the tcl/tk min and max scalar values

    minSV.set( int( min ) );
    maxSV.set( int( max ) );
  }
    
  return 1;
}



/**************************************************************
 *
 * Parses the strings x, y which are lists of integers
 * corresponding to nodes of the transfer function.
 *
 **************************************************************/

void
VolVis::UpdateTransferFncArray( clString x, clString y )
{
  int i, len, position;
  char * array;
  char * suppl = new char[CANVAS_WIDTH*4+1];
  char * form = new char[64];

  // clear the scalar value and opacity arrays

  Opacity.remove_all();
  ScalarVal.remove_all();

  // read in an integer and store the rest of the string
  // in suppl
  
  sprintf( form, "%%d %%%dc", CANVAS_WIDTH*4+1 );

  // clear suppl

  for ( i = 0; i < CANVAS_WIDTH*4+1; i++ )
    suppl[i] = '\0';
  
  /* read in the x-position */

  array = x();
  NodeCount = 0;
  len   = 1;

  while ( len )
    {
      sscanf( array, form, &position, suppl );
      strcpy( array, suppl );

      len = strlen( suppl );

      // clear the last few digits of suppl because %c does
      // not attach a null character

      if ( len - 5 > 0 )
	for ( i = len - 1; i >= len - 5; i-- )
	  suppl[i] = '\0';
      else
	for ( i = len - 1; i >= 0; i-- )
	  suppl[i] = '\0';

      ScalarVal.add( 1.0 * position / CANVAS_WIDTH *
	( maxSV.get() - minSV.get() ) + minSV.get() );
      
      Xvalues[NodeCount++] = 1.0 * position / CANVAS_WIDTH *
	( maxSV.get() - minSV.get() ) + minSV.get();

    }

  /* read in the y-position */

  array = y();
  NodeCount = 0;
  len   = 1;

  while ( len )
    {
      sscanf( array, form, &position, suppl );
      strcpy( array, suppl );

      len = strlen( suppl );

      // clear the last few digits of suppl

      if ( len - 5 > 0 )
	for ( i = len - 1; i >= len - 5; i-- )
	  suppl[i] = '\0';
      else
	for ( i = len - 1; i >= 0; i-- )
	  suppl[i] = '\0';

      // in tcl, the y value increases as one moves down

      Opacity.add( 1.0 * ( CANVAS_WIDTH - position )
	/ ( CANVAS_WIDTH ) );

      Yvalues[NodeCount++] = 1.0 * ( CANVAS_WIDTH - position )
	/ ( CANVAS_WIDTH );
    }

#if 0  
  // check the 2 arrays (TEMP)

  cerr << "The arrays are #" << NodeCount << ":\n";
  
  for ( i = 0; i < NodeCount; i++ )
    {
      cerr << i << ": ( " << Xvalues[i] << ", " << Yvalues[i]
	<< " )\n";
    }

  cerr << "SV @# " << ScalarVal.size() << endl;
  for ( i = 0; i < ScalarVal.size(); i++ )
    cerr << i << " #@: " << ScalarVal[i] << endl;

  cerr << "Opacity @# " << Opacity.size() << endl;
  for ( i = 0; i < Opacity.size(); i++ )
    cerr << i << " #@: " << Opacity[i] << endl;
#endif  

}




/**************************************************************
 *
 * Initializes an OpenGL window.
 *
 **************************************************************/

int
VolVis::makeCurrent() {

  // TEMP!!!  this used to be in the class; but i didn't need
  // it!  so, i moved these declarations down here...

  Tk_Window tkwin;
  Window win;
  GLXContext cx;


  // lock a mutex
  TCLTask::lock();

  // associate this with the "USER INTERFACE" button
  clString myname(clString(".ui")+id+".gl.gl");

  // find the tk window token for the window
  tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));

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
  // this command prints "cx=some_number" something to the screen
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
  glViewport( 0, 0, ViewPort-1, ViewPort-1 );

  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  // Set up an orthogonal projection to the screen
  // of size ViewPort
  glOrtho(0, ViewPort-1, ViewPort-1, 0, -1, 1);
  
  glMatrixMode(GL_MODELVIEW);
  
  glLoadIdentity();
  
  return 1;
}



/**************************************************************
 *
 * redraws but does not recalculate everything
 *
 **************************************************************/

void
VolVis::redraw_all()
{
  if ( ! makeCurrent() ) return;

  // clear the GLwindow to background color

  glClearColor( bgColor.r(), bgColor.g(), bgColor.b(), 0 );
  
  glClear(GL_COLOR_BUFFER_BIT);
  
  // if the handle was empty, just flush the buffer (to clear the window)
  // and return.
  
  if (! homeSFHandle.get_rep())
    {
      glFlush();
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }

  // lock because Image will be used

  imagelock.lock();

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  glPixelZoom(x_pixel_size, y_pixel_size);
  
  glRasterPos2i( x_pixel_size, y_pixel_size * (rasterY+1) );
  
  Color c(1, 0, 1);

  char *pixels=(char *) ( &(Image(0,0)) );

  // make sure that the routine to draw pixels is given the raster y first.
  // it wants width, then height
  
  glDrawPixels( rasterX, rasterY, GL_RGB, GL_UNSIGNED_BYTE, pixels );

  imagelock.unlock();

  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);


  TCLTask::unlock();
}	    



/**************************************************************
 *
 * if the scalar field is valid, calculates the new image.
 *
 **************************************************************/

void
VolVis::execute()
{
  ScalarFieldRG *homeSFRGrid;
  ColormapHandle cmap;
  CharColor temp;
  Array2<CharColor> * tempImage;

  // execute if the input ports are valid and if it is necessary
  // to execute

  if ( ! Validate( &homeSFRGrid ) )
    return;

  cerr << "        EXECUTING\n";

  // retrieve the scalar value-opacity values, and store them
  // in an array

  UpdateTransferFncArray( Xarray.get(), Yarray.get() );

  // instantiate the levoy class

  Levoy levoyModule ( homeSFRGrid, colormapport,
		     ibgColor.get() * ( 1. / 255 ), ScalarVal, Opacity );

  // calculate the new image

  tempImage = levoyModule.TraceRays ( iView.get(),
			 iRasterX.get(), iRasterY.get(), projection.get() );

  // lock it because the Image array will be modified

  imagelock.lock();

  Image = *tempImage;

  // also, the bgColor accessed by the redraw_all fnc
  // can now be changed.

  bgColor = ibgColor.get() * ( 1. / 255 );
  rasterX = iRasterX.get();
  rasterY = iRasterY.get();

  // the Image array has been modified, it is now safe to let
  // go of the thread
  
  imagelock.unlock();

  delete tempImage;

  // execute a tcl command

  TCL::execute(id+" redraw_when_idle");
  
}




/**************************************************************
 *
 * processes requests initiated by tcl module code
 *
 **************************************************************/

void
VolVis::tcl_command(TCLArgs& args, void* userdata) {
  
  if ( args[1] == "redraw_all" )
      redraw_all();
  else if ( args[1] == "wanna_exec" )
    want_to_execute();
  else
      Module::tcl_command(args, userdata);
}


