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
#include <Datatypes/GeometryPort.h>

#include <Geom/Color.h>


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
#include <string.h>

#include <Classlib/Timer.h>

#include <Modules/Visualization/LevoyVis.h>
#include "kuswik.h"


// the initial view data

const View homeview
(Point(0.6, 2.6, 0.6), Point(0.6, 0.6, 0.6), Vector(1.,0.,0.), 30);

// tcl interpreter corresponding to this module

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
  // Geometry Output port - connected to Salmon
  GeometryOPort* ogeom;

  // scalar field input port, provides the 3D image data
  
  ScalarFieldIPort *iport;

  // color map input port, provides scalar field to color map
  // if non-existant, white is the color used

  ColormapIPort* colormapport;

  // handle to the scalar field
  
  ScalarFieldHandle homeSFHandle;

  // remember last generation of scalar field

  int homeSFgeneration;
  
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

  TCLdouble minSV, maxSV;

  double min, max;

  // a list of {x,y}positions {x=scalar value, y=opacity}
  // associated with the nodes of "Opacity map" widget
  
  TCLstring Xarray, Yarray;
  TCLstring Rsv, Rop;
  TCLstring Gsv, Gop;
  TCLstring Bsv, Bop;

  // projection type

  TCLint projection;

  // allignment

  TCLint iCenter;

  // parallel or single processor

  TCLint iProc;
  
  // background color
  
  Color bgColor;

  // arrays with scalar value -- opacity mapping
  
  Array1<double> Opacity[4];
  Array1<double> ScalarVal[4];
  
  // the number of nodes
  
  int       NodeCount;
  
  // the image is stored in this 2d-array.  it contains
  // pixel values arranged in the order row then column

  Array2<CharColor> Image;



  // points to the display

  Display* dpy;


  
  // mutex to prevent problems related to changes of the
  // Image array

  Mutex imagelock;


  // number of processors is necessary for proper division
  // of data during parallel processing

  int procCount;

  // pointer to the levoy structure
  
  Levoy * calc;

  // specifies the number of separate slices processed by
  // each of the processors

  TCLint intervalCount;

  // The following variables will be supplied by Salmon
  // if the Salmon module is connected to this VolVis module.
  
  // SalmonView  = camera info for the scene,
  // DepthBuffer = distance from the eye to some pt in space,
  // ColorBuffer = RGB(A?) background color for each pixel.

  View SalmonView;

  Array2<double> DepthBuffer;
  Array2<Color>  ColorBuffer;
  

  // the variable that tells me that the ui is open
  
  TCLint uiopen;
  
  // initialize an OpenGL window
  
  int makeCurrent();

  int Validate ( ScalarFieldRG **homeSFRGrid );

  // place the {x,y}positions supplied by tcl-lists
  // into the arrays.

  void UpdateTransferFncArray( clString x, clString y, int index );

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
  
  void parallel(int proc);
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
  minSV("minSV", id, this),
  maxSV("maxSV", id, this),
  ibgColor("bgColor", id, this),
  projection("project", id, this),
  iCenter("centerit", id, this),
  iRasterX("rasterX", id, this),
  iRasterY("rasterY", id, this),
  iProc("processors", id, this),
  intervalCount("intervalCount", id, this),
  Rsv("Rsv", id, this),
  Gsv("Gsv", id, this),
  Bsv("Bsv", id, this),
  Rop("Rop", id, this),
  Gop("Gop", id, this),
  Bop("Bop", id, this),
  uiopen("uiopen", id, this),
  Xarray("Xarray", id, this),
  Yarray("Yarray", id, this)
{
  // Create a Colormap input port
  
  colormapport=scinew ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
  add_iport(colormapport);

  // Create a ScalarField input port
  
  iport = scinew ScalarFieldIPort(this, "RGScalarField", ScalarFieldIPort::Atomic);
  add_iport(iport);

  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);

  // initialize the view to home view

  iView.set( homeview );

  // initialize a few variables used by OpenGL
  
  x_pixel_size = 1;
  y_pixel_size = 1;

  homeSFgeneration = -1;

  bgColor = BLACK;

  // initialize this image in order to prevent seg faults
  // by OpenGL when redrawing the screen with uninitialized
  // (size of 0,0) array of pixels

  Image.newsize(100,100);
  Image.initialize( bgColor );

  // TEMP: i have no clue why initialization of ibgColor in
  // TCL code does not work.

  ibgColor.set(BLACK);

  //  intervalCount.set(1);

  max = 110;
  min = 1;
}



/**************************************************************
 *
 * copy constructor
 *
 **************************************************************/

VolVis::VolVis(const VolVis& copy, int deep)
: Module(copy, deep),
  iView("View", id, this),
  maxSV("maxSV", id, this),
  minSV("minSV", id, this),
  iRasterX("rasterX", id, this),
  iRasterY("rasterY", id, this),
  ibgColor("bgColor", id, this),
  projection("project", id, this),
  iCenter("centerit", id, this),
  iProc("processors", id, this),
  intervalCount("intervalCount", id, this),
  Rsv("Rsv", id, this),
  Gsv("Gsv", id, this),
  Bsv("Bsv", id, this),
  Rop("Rop", id, this),
  Gop("Gop", id, this),
  Bop("Bop", id, this),
  uiopen("uiopen", id, this),
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
 *
 *
 **************************************************************/

static void
do_parallel(void* obj, int proc)
{
  VolVis* module=(VolVis*)obj;
  module->parallel(proc);
}





/**************************************************************
 *
 *
 *
 **************************************************************/

void
VolVis::parallel( int proc )
{
  int i;

  int interval = iRasterX.get() / procCount / intervalCount.get();
cerr << intervalCount.get();

  if ( projection.get() )
    {
      if ( proc != procCount - 1 )
	for ( i = 0; i < intervalCount.get(); i++ )
	  {
	    calc->PerspectiveTrace( interval * ( proc + procCount * i ),
				   interval * ( proc + 1 + procCount * i ) );
//	    cout << "between: " << interval * (proc + procCount * i ) << "  "
//	      << interval * (proc + 1 + procCount * i ) << endl;
	  }
      
      else
	{
	  for ( i = 0; i < intervalCount.get()-1; i++ )
	    {
	    calc->PerspectiveTrace( interval * ( proc + procCount * i ),
				   interval * ( proc + 1 + procCount * i ) );
//	    cout << "between: " << interval * (proc + procCount * i ) << "  "
//	      << interval * (proc + 1 + procCount * i ) << endl;
	  }
	  
	  calc->PerspectiveTrace(interval * ( proc + procCount *
					      ( intervalCount.get() - 1 ) ),
				 iRasterX.get() );
//	    cout << "between: " << interval * (proc + procCount *
//					       (intervalCount.get() - 1))
//	      << "  " << iRasterX.get() << endl;
	}
    }
  else
    {
      if ( proc != procCount - 1 )
	for ( i = 0; i < intervalCount.get(); i++ )
	  calc->ParallelTrace( interval * ( proc + procCount * i ),
			      interval * ( proc + 1 + procCount * i ) );
      else
	{
	  for ( i = 0; i < intervalCount.get()-1; i++ )
	    calc->ParallelTrace( interval * ( proc + procCount * i ),
				interval * ( proc + 1 + procCount * i ) );
	  
	  calc->ParallelTrace( interval * ( proc + procCount *
					   ( intervalCount.get() - 1 ) ),
			      iRasterX.get() );
	}
    }
}


#if 0
void
VolVis::parallel( int proc )
{
  int i;

  int interval = iRasterX.get() / procCount / intervalCount.get();


  if ( projection.get() )
    {
      if ( proc != procCount - 1 )
	for ( i = 0; i < intervalCount.get(); i++ )
	  {
	    cerr << proc << "!!! " << interval * (proc + procCount * i ) << "  ";
	    cerr << interval * (proc + 1 + procCount * i ) << endl;
	    
	    calc->PerspectiveTrace( interval * ( proc + procCount * i ),
				   interval * ( proc + 1 + procCount * i ) );
	  }
      else
	{
	  for ( i = 0; i < intervalCount.get()-1; i++ )
	    {
	      cerr << proc << "!!! " << interval * (proc + procCount * i ) << "  ";
	      cerr << interval * (proc + 1 + procCount * i ) << endl;
	      
	      calc->PerspectiveTrace( interval * ( proc + procCount * i ),
				     interval * ( proc + 1 + procCount * i ) );
	    }
	  
	  cerr << proc << "!!! " << interval * (proc + procCount * i ) << "  ";
	  cerr << interval * (proc + 1 + procCount * i ) << endl;
	  
	  calc->PerspectiveTrace( interval * ( proc + procCount *
					      ( intervalCount.get() - 1 ) ),
				 iRasterX.get() );
	}
    }
  else
    {
      if ( proc != procCount - 1 )
	for ( i = 0; i < intervalCount.get(); i++ )
	  calc->ParallelTrace( interval * ( proc + procCount * i ),
			      interval * ( proc + 1 + procCount * i ) );
      else
	{
	  for ( i = 0; i < intervalCount.get()-1; i++ )
	    calc->ParallelTrace( interval * ( proc + procCount * i ),
				interval * ( proc + 1 + procCount * i ) );
	  
	  calc->ParallelTrace( interval * ( proc + procCount *
					   ( intervalCount.get() - 1 ) ),
			      iRasterX.get() );
	}
    }
}
#endif


/**************************************************************
 *
 * make sure that the scalar field handle points to valid
 * data.  initialize some variables.
 *
 **************************************************************/

int
VolVis::Validate ( ScalarFieldRG **homeSFRGrid )
{
  cerr << "Validate\n";
  
  // get the scalar field handle
  
  if ( ! iport->get(homeSFHandle) )
    cerr << "\n\n\n\n\n\n\n\niport->get the handle returned FALSE\n\n\n\n\n\n\n\n\n\n";

  // Make sure this is a valid scalar field

  if ( ! homeSFHandle.get_rep() )
    {
      cerr << "bailed on getting the representation\n";
      return 0;
    }

  // make sure scalar field is a regular grid
  // actually, it doesn't have to be (Steve says so)
  
  if ( ( (*homeSFRGrid) = homeSFHandle->getRG()) == 0)
    {
      cerr << "bailed because it is not a regular grid\n";
      return 0;
    }

  if ( homeSFgeneration != homeSFHandle->generation )
    {
      // remember the generation
      
      homeSFgeneration = homeSFHandle->generation;

      // get the min, max values of the new field


      (*homeSFRGrid)->get_minmax( min, max );

      // set the tcl/tk min and max scalar values

      minSV.set( min );
      maxSV.set( max );
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
VolVis::UpdateTransferFncArray( clString x, clString y, int index )
{
  int i, len, position;
  char * array;
  char * suppl = new char[CANVAS_WIDTH*4+1];
  char * form = new char[64];

  // clear the scalar value and opacity arrays

  Opacity[index].remove_all();
  ScalarVal[index].remove_all();

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

      ScalarVal[index].add( 1.0 * position / CANVAS_WIDTH *
			   ( max - min ) + min );

//      cerr << index << " it is: " << 1.0 * position / CANVAS_WIDTH * (max-min) + min << endl;
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

      Opacity[index].add( 1.0 * ( CANVAS_WIDTH - position )
			 / ( CANVAS_WIDTH ) );
    }

}




/**************************************************************
 *
 * Initializes an OpenGL window.
 *
 **************************************************************/

int
VolVis::makeCurrent()
{
  cerr << "Made current in " << Task::self()->get_name() << endl;

  // several variables used in this function
  
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
  glViewport( 0, 0, VIEW_PORT_SIZE-1, VIEW_PORT_SIZE-1 );

  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  // Set up an orthogonal projection to the screen
  // of size VIEW_PORT_SIZE
  glOrtho(0, VIEW_PORT_SIZE-1, VIEW_PORT_SIZE-1, 0, -1, 1);
  
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

  if ( ! uiopen.get() )
    return;
  
  if ( ! makeCurrent() )
      return;

  // clear the GLwindow to background color

  glClearColor( bgColor.r(), bgColor.g(), bgColor.b(), 0 );
  
  glClear(GL_COLOR_BUFFER_BIT);
  
  // if the handle was empty, just flush the buffer (to clear the window)
  // and return.

  if (! homeSFHandle.get_rep())
    {
      glFlush();
      cerr << "Made uncurrent #2\n";
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }

  // lock because Image will be used

  imagelock.lock();

  // initialize some OpenGL variables

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  glPixelZoom(x_pixel_size, y_pixel_size);

//  cerr << "Displaying an image of dimensions: " << Image.dim1() << " " <<
//    Image.dim2() << endl;
  
  if ( iCenter.get() )
    {
      glRasterPos2i( x_pixel_size + ( VIEW_PORT_SIZE - Image.dim2() ) / 2 ,
		    y_pixel_size * ( ( VIEW_PORT_SIZE / 2 + Image.dim1() / 2 + 1 ) ) );
    }
  else
    {
      glRasterPos2i( x_pixel_size, y_pixel_size * (Image.dim1()+1) );
    }
  

  Color c(1, 0, 1);

  char *pixels=(char *) ( &(Image(0,0)) );

  // glDrawPixles wants width, then height
  
  glDrawPixels( Image.dim2(), Image.dim1(), GL_RGB, GL_UNSIGNED_BYTE, pixels );

  imagelock.unlock();

  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }

  cerr << "Made uncurrent..." << endl;
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
  WallClockTimer watch;
  
  // make sure TCL variables are updated
  
  ScalarFieldRG *homeSFRGrid;

  // execute if the input ports are valid and if it is necessary
  // to execute

  if ( ! Validate( &homeSFRGrid ) )
    return;

  // update the values from TCL interface

  reset_vars();

  // continue if UI is open
  
  if ( ! uiopen.get() )
    return;
  
  // get data from transfer map and make sure these changes
  // are reflected in the TCL variables

  TCL::execute(id+" get_data");
  reset_vars();
  

  // retrieve the scalar value-opacity/rgb values, and store them
  // in arrays

  UpdateTransferFncArray( Xarray.get(), Yarray.get(), 0 );
  UpdateTransferFncArray( Rsv.get(), Rop.get(), 1 );
  UpdateTransferFncArray( Gsv.get(), Gop.get(), 2 );
  UpdateTransferFncArray( Bsv.get(), Bop.get(), 3 );

  // if connected to a Salmon module, retrieve Salmon geometry info
  // and store it in a LevoyS type structure
  // otherwise, use the Levoy structure

  GeometryData* data=ogeom->getData(0, GEOM_ALLDATA);

  if ( data != NULL )
    {
      calc = new LevoyS( homeSFRGrid, colormapport,
			ibgColor.get() * ( 1. / 255 ), ScalarVal, Opacity );

      calc->SetUp( data );
    }
  else
    {
      calc = new Levoy ( homeSFRGrid, colormapport,
			ibgColor.get() * ( 1. / 255 ), ScalarVal, Opacity );

      calc->SetUp( iView.get(), iRasterX.get(), iRasterY.get() );
    }      

  watch.start();

  cerr << "projection is: " << projection.get() << " interval count is " <<
    intervalCount.get() <<endl;
  cerr << "raster size: " << iRasterX.get() << endl;
  
  if ( intervalCount.get() != 0 && iProc.get() )
    {
      procCount = Task::nprocessors();


      Task::multiprocess(procCount, do_parallel, this);
    }
  else
    calc->TraceRays( projection.get() );

  watch.stop();

  cerr << "my watch reports: " << watch.time() << "units of time\n";

  // lock it because the Image array will be modified

  imagelock.lock();

  Image = *(calc->Image);

  /* THE MOST AWESOME DEBUGGING TECHNIQUE */
  
  int loop;

  int d1, d2;
  d1 = Image.dim1() - 1;
  d2 = Image.dim2();
  
  for ( loop = 0; loop < d2; loop++ )
    {
      Image(0, loop).red = 255;
      Image(0, loop).green = 0;
      Image(0, loop).blue = 0;
      
      Image(d1, loop).red = 255;
      Image(d1, loop).green = 0;
      Image(d1, loop).blue = 0;
    }
  
  int pool;
  d1++; d2--;
  
  for ( pool = 0; pool < d1; pool++ )
    {
      Image(pool, 0).red = 0;
      Image(pool, 0).green = 0;
      Image(pool, 0).blue = 255;
      
      Image(pool, d2).red = 0;
      Image(pool, d2).green = 0;
      Image(pool, d2).blue = 255;
    }
  
  /* END OF THE MOST AWESOME DEBUGGING TECHNIQUE */
  
  // also, the bgColor accessed by the redraw_all fnc
  // can now be changed.

  bgColor = ibgColor.get() * ( 1. / 255 );

  // the Image array has been modified, it is now safe to let
  // go of the thread
  
  imagelock.unlock();


  // TEMP: execute a tcl command (what does this do???)
  
  update_progress(0.5);

  TCL::execute(id+" redraw_when_idle");

  delete calc;
}




/**************************************************************
 *
 * processes requests initiated by tcl module code
 *
 **************************************************************/

void
VolVis::tcl_command(TCLArgs& args, void* userdata)
{
  if ( args[1] == "redraw_all" ){
    if(strcmp(Task::self()->get_name(), "TCLTask"))
      {
	TCL::execute("after idle "+args[0]+" "+args[1]);
	cerr << "Attempted redraw from " << Task::self()->get_name()
	  << ", deferred...\n";
      }
    else
	redraw_all();
  }
  else if ( args[1] == "wanna_exec" )
      want_to_execute();
  else
    Module::tcl_command(args, userdata);
}
