//static char *id="@(#) $Id$";

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
#include <CoreDatatypes/ScalarFieldRG.h>
#include <CommonDatatypes/ScalarFieldPort.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/ColorMapPort.h>

#include <Geometry/BBox.h>
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
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCL.h>

#include <tcl.h>
#include <tk.h>

#include <Containers/Array2.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include <Util/Timer.h>

#include <Multitask/Mailbox.h>
#include <Multitask/Task.h>
#include <Multitask/ITC.h>

#include <Math/Trig.h>

#include <Modules/Visualization/FastRender.h>
#include "kuswik.h"


#define NEWSCALARFIELD   1
#define NEWVIEW          2
#define NEWTRANSFERMAP   3
#define NEWRASTER        4
#define NEWSTEP          5
#define NEWINVALIDSCALARFIELD   6
#define NEWBACKGROUND    7
#define NEWLINEARATTENUATION 8
#define NEWColorMap      9

#define VTABLE_SIZE      2000

#define VIEW_PORT_SIZE 600
#define CANVAS_WIDTH 200
#define TABLE_SIZE 2000

// tcl interpreter corresponding to this module

extern Tcl_Interp* the_interp;

// the OpenGL context structure

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;

typedef void (*FR) ( double, double, double, double, double, double, double,
		    double *, double *, double *, double *, double, double,
		    double, double, double, double ***grid, double, double, double,
		    int, int, int, double, double, double, double,
		    double *, double *, double * );

class RenderingThread;
class VolVis;

class RenderingThread : public Task
{
public:
  VolVis *v;
  RenderingThread( VolVis * voldata );
  int body( int );
};


RenderingThread::RenderingThread( VolVis *voldata ) : Task( "VVRenderLoop" )
{
  v = voldata;
}


// TEMP! make the parallel fnc pretty by not duplicating the same
// TEMP! procedure for the perspective and orthogonal projections

// the initial view data

const ExtendedView ehomeview
(Point(0.6, 0.5, -0.7), Point(0.5, 0.5, 0.5), Vector(0.,1.,0.), 60, 100, 100,
 Color(0.,0.,0.));

//typedef Mailbox<SimplePortComm<int>*> SMail;

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

  // handle to the scalar field
  ScalarFieldHandle homeSFHandle;

  // PETE
  ColorMapHandle cmap;

  // remember last generation of scalar field
  int homeSFgeneration;
  
  // the size in pixels of output window
  double x_pixel_size;
  double y_pixel_size;

  
  // try the extended view stuff
  TCLExtendedView iEView;
  ExtendedView myview;

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
  TCLint Petes;

  // parallel or single processor
  TCLint iProc;
  
  // background color
  Color bgColor;

  // arrays with scalar value -- opacity mapping
  Array1<double> AssociatedVals[4];
  Array1<double> ScalarVals[4];
  
  // the image is stored in this 2d-array.  it contains
  // pixel values arranged in the order row then column
  Array2<CharColor> Image;



  // points to the display
  Display* dpy;

  Window   win;

  
  // mutex to prevent problems related to changes of the
  // Image array
  Mutex imagelock;
  Mutex donelock;

  // number of processors is necessary for proper division
  // of data during parallel processing
  int procCount;

  // pointer to the levoy structure
//  Levoy calc;

  // specifies the number of separate slices processed by
  // each of the processors
  TCLint intervalCount;

  
  // the variable that tells me that the ui is open
  TCLint uiopen;

  int already_open;


  // if Salmon is connected to VolVis, the user can chose to:
  //    &* not use any of the data provided by Salmon
  //    &* use just the view information
  //    &* use all the information: view, z, and rgb values
  TCLint salmonData;

  // allow the user to specify the step size for Levoy's algorithm
  TCLint stepSize;
  
  //
  //
  //
  // initialize an OpenGL window
  //
  
  int makeCurrent();

#if 0  
  int Validate ( ScalarFieldRG **homeSFRGrid );
#endif  

  // place the {x,y}positions supplied by tcl-lists
  // into the arrays.
  void UpdateTransferFncArray( clString x, clString y, int index );

  static const int numMsg=100;
  
  //  SMail msgs;
  Mailbox<int> msgs;

  RenderingThread *rt;
  
  void XRasterChanged( clString a );
  void StepSizeChanged( clString a );
  void TransferMapChanged();

  ScalarFieldRG * homeSFRGrid;
  int stepcount;

  // stuff for the transfer map
  int SVMultiplier;
  double *SVOpacity;
  double *SVR;
  double *SVG;
  double *SVB;
  
  double *SVOpacitysave;
  double *SVRsave;
  double *SVGsave;
  double *SVBsave;
  
  Array1<double> Slopes[4];

  unsigned int mask;
  int whiteFlag;
  BBox box;
  Point bmin, bmax;
  
  Vector homeRay;
  Vector rayIncrementU, rayIncrementV;
  double rayStep;

  int nx, ny, nz;
  Vector diagonal;
  double diagx, diagy, diagz;
  
  
  double DetermineRayStepSize ( int steps );
  void CalculateRayIncrements ();
  void CreateTransferTables();
  void ScalarFieldChanged();
  void UTArrays();

  void ViewChanged( clString a );
  unsigned int FindMask( unsigned int seq );

  unsigned int ONE;
  int xres, yres;
  int width;

  Point eye;

  unsigned int global_start_seq, global_seq, new_seq;
  int done;
  int xmiddle, ymiddle;

  int xresChanged, yresChanged, stepcountChanged;
  ScalarFieldRG * data;
  
  Color newbg;
  void RBackgroundChanged( clString a );
  void GBackgroundChanged( clString a );
  void BBackgroundChanged( clString a );

  View newview;

  FR CastRay;

  double mbgr, mbgg, mbgb;
  double newLinearA, LinAtten;

  void LinearAChanged( clString a );
  ColorMapIPort* inColorMap;

  void ReadTransfermap();

  int jelly;

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

  void RenderLoop();
};




/**************************************************************
 *
 * TEMP!
 *
 **************************************************************/



/**************************************************************
 *
 *
 *
 **************************************************************/

Module* make_VolVis(const clString& id) {
  return new VolVis(id);
}




/**************************************************************
 *
 * constructor
 *
 **************************************************************/

VolVis::VolVis(const clString& id)
: Module("VolVis", id, Source),
  iEView("eview", id, this),
  minSV("minSV", id, this),
  maxSV("maxSV", id, this),
  projection("project", id, this),
  Petes("pcmap", id, this),
  iProc("processors", id, this),
  intervalCount("intervalCount", id, this),
  Rsv("Rsv", id, this),
  Gsv("Gsv", id, this),
  Bsv("Bsv", id, this),
  Rop("Rop", id, this),
  Gop("Gop", id, this),
  Bop("Bop", id, this),
  uiopen("uiopen", id, this),
  salmonData("salmon", id, this),
  stepSize("stepsize", id, this),
  Xarray("Xarray", id, this),
  Yarray("Yarray", id, this),
  msgs(10)
{
  cerr << "Welcome to VolVis\n";
  
  // Create a ScalarField input port
  
  iport = scinew ScalarFieldIPort(this, "RGScalarField", ScalarFieldIPort::Atomic);
  add_iport(iport);

  inColorMap=scinew  
    ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
  
  add_iport(inColorMap);
					
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);

  // initialize the view to home view
  iEView.set( ehomeview );
  myview = ehomeview;

  eye = myview.eyep();
  homeRay = myview.lookat() - eye;
  homeRay.normalize();
  CalculateRayIncrements();
  rayStep = 1.0; // just for kicks, it will have to be recalculated
  stepcount = 8;
  

  // initialize a few variables used by OpenGL
  x_pixel_size = 1;
  y_pixel_size = 1;

  
  homeSFgeneration = -1;

  // initialize this image in order to prevent seg faults
  // by OpenGL when redrawing the screen with uninitialized
  // (size of 0,0) array of pixels

  Image.newsize(600,600);
  Color bg(myview.bg());
  Image.initialize( bg );

  iEView.bg.set( BLACK );

  min = 1; max = 110;
  already_open = 0;
  homeSFRGrid = NULL;
  rt = NULL;
  SVMultiplier = 1.0;
  mask = 0x3500;
  whiteFlag = 1;
  bmin.x( 0.0 );
  bmin.y( 0.0 );
  bmin.z( 0.0 );
  bmax.x( 1.0 );
  bmax.y( 1.0 );
  bmax.z( 1.0 );
  diagonal.x( 1.0 );
  diagonal.y( 1.0 );
  diagonal.z( 1.0 );
  diagx = diagy = diagz = 1.0;
  nx = ny = nz = 100;
  ONE = 0x0001;
  xres = yres = 100;
  xresChanged = yresChanged = 100;
  mbgr = mbgg = mbgb = 0.0;
  
  // create and activate the rendering thread
  rt = new RenderingThread( this );
  rt->activate( 0 );
  cerr << "Activated the thread\n";

  CastRay = &BasicVolRender;
  LinAtten = 1;

  SVOpacitysave = new double[VTABLE_SIZE];
  SVRsave = new double[VTABLE_SIZE];
  SVGsave = new double[VTABLE_SIZE];
  SVBsave = new double[VTABLE_SIZE];
  SVOpacity = SVOpacitysave;
  SVR = SVRsave;
  SVG = SVGsave;
  SVB = SVBsave;
}



/**************************************************************
 *
 * copy constructor
 *
 **************************************************************/

VolVis::VolVis(const VolVis& copy, int deep)
: Module(copy, deep),
  iEView("eview", id, this),
  maxSV("maxSV", id, this),
  minSV("minSV", id, this),
  projection("project", id, this),
  Petes("pcmap", id, this),
  iProc("processors", id, this),
  intervalCount("intervalCount", id, this),
  Rsv("Rsv", id, this),
  Gsv("Gsv", id, this),
  Bsv("Bsv", id, this),
  Rop("Rop", id, this),
  Gop("Gop", id, this),
  Bop("Bop", id, this),
  uiopen("uiopen", id, this),
  salmonData("salmon", id, this),
  stepSize("stepsize", id, this),
  Xarray("Xarray", id, this),
  Yarray("Yarray", id, this),
  msgs(10)
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



void
VolVis::parallel( int proc )
{
  int i,j;
  int seq = global_seq;
  double r, g, b;
  Vector rayToTrace;
  Point beg;
  int row, col;

  for ( i = 0; i < proc; i++ )
    {
      // compute the next sequence element
      if ( (ONE & seq) == ONE )
	seq = (seq>>1) ^ ( mask );
      else
	seq = seq>>1;

      if ( seq == global_start_seq )
	{
	  done = 1;
	  return;
	}
    }

  for ( i = 0; i < 2000; i++ )
    {
      row = seq / xres;
      col = seq % xres;
      
      if ( row < yres )
	{
	  // CAST A RAY
	  
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( col - xmiddle );
	  rayToTrace += rayIncrementV * ( row - ymiddle );
	  rayToTrace.normalize();
	  
	  // make the ray the length of the step
	  rayToTrace *= rayStep;
	  
	  // must add the offset(to put the image in the middle)
	  
	  if ( box.Intersect( eye, rayToTrace, beg ) )
	    {
	      (*CastRay)( rayToTrace.x(), rayToTrace.y(),
			 rayToTrace.z(), rayStep, beg.x(),
			 beg.y(), beg.z(), SVOpacity,
			 SVR, SVG, SVB, min,
			 SVMultiplier, box.min().x(),
			 box.min().y(), box.min().z(),
			 homeSFRGrid->grid.get_dataptr(),
			 mbgr, mbgg, mbgb, nx, ny, nz,
			 diagx, diagy, diagz, LinAtten,
			 &r, &g, &b );
	      
	      Image( row, col ).red   = (unsigned char)(r*255.0);
	      Image( row, col ).green = (unsigned char)(g*255.0);
	      Image( row, col ).blue  = (unsigned char)(b*255.0);
	    }
	  else
	    {
	      Image( row, col ).red   = (unsigned char)(mbgr*255.0);
	      Image( row, col ).green = (unsigned char)(mbgg*255.0);
	      Image( row, col ).blue  = (unsigned char)(mbgb*255.0);
	    }
	}

      for ( j = 0; j < procCount; j++ )
	{
	  // compute the next sequence element
	  if ( (ONE & seq) == ONE )
	    seq = (seq>>1) ^ ( mask );
	  else
	    seq = seq>>1;

	  if ( seq == global_start_seq )
	    {
	      done = 1;
	      return;
	    }
	}
    }

      row = seq / xres;
      col = seq % xres;
      
      if ( row < yres )
	{
	  // CAST A RAY
	  
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( col - xmiddle );
	  rayToTrace += rayIncrementV * ( row - ymiddle );
	  rayToTrace.normalize();
	  
	  // make the ray the length of the step
	  rayToTrace *= rayStep;
	  
	  // must add the offset(to put the image in the middle)
	  
	  if ( box.Intersect( eye, rayToTrace, beg ) )
	    {
	      (*CastRay)( rayToTrace.x(), rayToTrace.y(),
			 rayToTrace.z(), rayStep, beg.x(),
			 beg.y(), beg.z(), SVOpacity,
			 SVR, SVG, SVB, min,
			 SVMultiplier, box.min().x(),
			 box.min().y(), box.min().z(),
			 homeSFRGrid->grid.get_dataptr(),
			 mbgr, mbgg, mbgb, nx, ny, nz,
			 diagx, diagy, diagz, LinAtten,
			 &r, &g, &b );
	      
	      Image( row, col ).red   = (unsigned char)(r*255.0);
	      Image( row, col ).green = (unsigned char)(g*255.0);
	      Image( row, col ).blue  = (unsigned char)(b*255.0);
	    }
	  else
	    {
	      Image( row, col ).red   = (unsigned char)(mbgr*255.0);
	      Image( row, col ).green = (unsigned char)(mbgg*255.0);
	      Image( row, col ).blue  = (unsigned char)(mbgb*255.0);
	    }
	}

  if ( proc == procCount - 1 )
    new_seq = seq;
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
  
  char *array = new char[4000];
  
  char * suppl = new char[CANVAS_WIDTH*4+1];
  char * form = new char[64];

  // clear the scalar value and opacity arrays

  AssociatedVals[index].remove_all();
  ScalarVals[index].remove_all();

  // read in an integer and store the rest of the string
  // in suppl
  
  sprintf( form, "%%d %%%dc", CANVAS_WIDTH*4+1 );

  // clear suppl

  for ( i = 0; i < CANVAS_WIDTH*4+1; i++ )
    suppl[i] = '\0';
  
  /* read in the x-position */

  strcpy( array, x());
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

      ScalarVals[index].add( 1.0 * position / CANVAS_WIDTH *
			   ( max - min ) + min );

//      cerr << index << " it is: " << 1.0 * position / CANVAS_WIDTH * (max-min) + min << endl;
    }

//  cerr << "The string is: " << x();

  /* read in the y-position */

  strcpy( array, y());
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

      AssociatedVals[index].add( 1.0 * ( CANVAS_WIDTH - position )
			 / ( CANVAS_WIDTH ) );
    }
//  cerr << "\nThe other string is " << y() << endl;
}




/**************************************************************
 *
 * Initializes an OpenGL window.
 *
 **************************************************************/

int
VolVis::makeCurrent()
{
//  cerr << "Made current in " << Task::self()->get_name() << endl;

  // several variables used in this function
  
  Tk_Window tkwin;

  GLXContext cx;


  // lock a mutex
  TCLTask::lock();
  
  // associate this with the "USER INTERFACE" button
  clString myname(clString(".ui")+id+".gl.gl");

  // find the tk window token for the window
  tkwin=Tk_NameToWindow(the_interp,
			const_cast<char *>(myname()),
			Tk_MainWindow(the_interp));

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
  cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));

  // check if it was created
  if(!cx)
    {
      cerr << "Unable to create OpenGL Context!\n";
      TCLTask::unlock();
      return 0;
    }

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
//  cerr << "Welcome to redraw_all\n";

//  if ( ! uiopen.get() )
//    return;
  
  if ( ! makeCurrent() )
      return;

  glDrawBuffer(GL_BACK);
  
  // clear the GLwindow to background color

  glClearColor( myview.bg().r(), myview.bg().g(), myview.bg().b(), 0 );
  
  glClear(GL_COLOR_BUFFER_BIT);
  
  // if the handle was empty, just flush the buffer (to clear the window)
  // and return.

  if (! homeSFHandle.get_rep())
    {
      glFlush();
//      cerr << "Made uncurrent #2\n";
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }

  // lock because Image will be used

//  imagelock.lock();

  // initialize some OpenGL variables

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  // set pixelsize based on other stuff...

  int scale_int = yres;
  if (xres > scale_int)
    scale_int = xres;

  float zoom = VIEW_PORT_SIZE*1.0/scale_int;

//  zoom = 1.0;
  x_pixel_size = 1.0*zoom;
  y_pixel_size = 1.0*zoom;

  glPixelZoom(x_pixel_size, y_pixel_size);

//  printf("%lf %lf : %lf : %d\n",x_pixel_size,y_pixel_size,zoom,scale_int);
  
  glRasterPos2i( 0, 599 );
  //      glRasterPos2i( x_pixel_size, y_pixel_size * (Image.dim1()+1) );
  

  //Color c(1, 0, 1);

  char *pixels=(char *) ( &(Image(0,0)) );

  // glDrawPixles wants width, then height
  
  glDrawPixels( Image.dim2(), Image.dim1(), GL_RGB, GL_UNSIGNED_BYTE, pixels );

  cerr << "in the swap buffer\n";
  glXSwapBuffers(dpy,win);
  
//  imagelock.unlock();
  
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }

//  cerr << "Made uncurrent..." << endl;
  glXMakeCurrent(dpy, None, NULL);

  TCLTask::unlock();


}


void
VolVis::RBackgroundChanged( clString a )
{
  double d;
  a.get_double( d );

  newbg.r( d );

  msgs.send( NEWBACKGROUND );
}


void
VolVis::GBackgroundChanged( clString a )
{
  double d;
  a.get_double( d );

  newbg.g( d );

  msgs.send( NEWBACKGROUND );
}


void
VolVis::BBackgroundChanged( clString a )
{
  double d;
  a.get_double( d );

  newbg.b( d );

  msgs.send( NEWBACKGROUND );
}



/**************************************************************
 *
 * if the scalar field is valid, calculates the new image.
 *
 **************************************************************/

void
VolVis::ScalarFieldChanged()
{
      // send a message about the new scalar field only if you can
      // retrieve a handle to it
      if ( ! iport->get( homeSFHandle ) )
	{
	  msgs.send( NEWINVALIDSCALARFIELD );
	  return;
	}
      
      // make sure this is a valid scalar field
      if ( ! homeSFHandle.get_rep() )
	{
	  msgs.send( NEWINVALIDSCALARFIELD );
	  return;
	}

      // make sure this is a valid regular grid
      if ( ( data = homeSFHandle->getRG() ) )
	msgs.send( NEWSCALARFIELD );
      else
	msgs.send( NEWINVALIDSCALARFIELD );

    }
      
void
VolVis::execute()
{
  if ( already_open )
    ScalarFieldChanged();

  if ( jelly )
    {
      inColorMap->get( cmap );
      cerr<<"jeller\n";
      if ( cmap.get_rep() )
	msgs.send( NEWColorMap );
      
      jelly = 0;
    }
  return;
}


void
VolVis::ReadTransfermap()
{
  msgs.send( NEWTRANSFERMAP );
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
  else if ( args[1] == "XRasterChanged" )
    XRasterChanged( args[2] );
  else if ( args[1] == "TMChanged" )
    TransferMapChanged();
  else if ( args[1] == "SSChanged" )
    StepSizeChanged( args[2] );
  else if ( args[1] == "UIOpen" )
    UTArrays();
  else if ( args[1] == "ControlButtonChanged" )
    {
      reset_vars();
      ViewChanged( args[1] );
    }
  else if ( args[1] == "ViewChanged" )
    ViewChanged( args[2] );
  else if ( args[1] == "RBackgroundChanged" )
    RBackgroundChanged( args[2] );
  else if ( args[1] == "GBackgroundChanged" )
    GBackgroundChanged( args[2] );
  else if ( args[1] == "BBackgroundChanged" )
    BBackgroundChanged( args[2] );
  else if ( args[1] == "LinearAChanged" )
    LinearAChanged( args[2] );
  else if ( args[1] == "ReadColorMap" )
    {
      cerr << "JOYOUOUOU\n";
      jelly = 1;
      want_to_execute();
    }
  else if ( args[1] == "ReadTransfermap" )
    ReadTransfermap();
  else
    Module::tcl_command(args, userdata);
}

void
VolVis::UTArrays()
{
      int i,j;
      reset_vars();
      already_open = 1;

      imagelock.lock();
      // retrieve the scalar value-opacity/rgb values, and store them
      // in arrays
      UpdateTransferFncArray( Xarray.get(), Yarray.get(), 0 );
      UpdateTransferFncArray( Rsv.get(), Rop.get(), 1 );
      UpdateTransferFncArray( Gsv.get(), Gop.get(), 2 );
      UpdateTransferFncArray( Bsv.get(), Bop.get(), 3 );

      for ( i = 0; i < 4; i++ )
	Slopes[i].remove_all();
      
      // calculate slopes between consecutive opacity values
      for ( i = 0; i < 4; i++ )
	Slopes[i].add( 0. );
      
      for ( i = 0; i < 4; i++ )
	for ( j = 1; j < ScalarVals[i].size(); j++ )
	  Slopes[i].add( ( AssociatedVals[i][j] - AssociatedVals[i][j-1] ) /
			( ScalarVals[i][j] - ScalarVals[i][j-1] ) );

//      cerr << "\n The following is the slopes array:\n";
      for ( i = 0; i < Slopes[0].size(); i++ )
//	cerr << i << ": " << Slopes[0][i] << endl;
      
      
      // if the rgb are at 1 for every scalar value, set the
      // whiteFlag to be true, otherwise it is false
      whiteFlag = TRUE;
      
      for ( i = 1; i < 4; i++ )
	{
	  if ( AssociatedVals[i].size() != 2 )
	    {
	      whiteFlag = FALSE;
	      break;
	    }
	  if ( AssociatedVals[i][0] != 1. || AssociatedVals[i][1] != 1. )
	    {
	      whiteFlag = FALSE;
	      break;
	    }
	}
      imagelock.unlock();
}

void
VolVis::XRasterChanged( clString a )
{
//  int r, mult;
//  a.get_int(r);

  a.get_int( xresChanged );

//  myview.xres( r );
//  xres = r;

  msgs.send( NEWRASTER );
}

unsigned int
VolVis::FindMask( unsigned int seq )
{
  unsigned int mult;

  imagelock.lock();
  mult = myview.yres() * myview.xres() -1;
  mult = mult >> 14;
  width = 14;
  while ( mult != 0 )
    {
      mult = mult>>1;
      width++;
    }
  switch( width )
    {
    case 14:
      mask = 0x3500;
      seq = 0x3fff & seq;
      break;
    case 15:
      mask = 0x6000;
      seq = 0x7fff & seq;
      break;
    case 16:
      mask = 0xB400;
      seq = 0xffff & seq;
      break;
    case 17:
      mask = 0x00012000;
      seq = 0x0001ffff & seq;
      break;
    case 18:
      mask = 0x00020400;
      seq = 0x0003ffff & seq;
      break;
    case 19:
      mask = 0x00072000;
      seq = 0x0007ffff & seq;
      break;
    case 20:
      mask = 0x00090000;
      break;
    }
  
  imagelock.unlock();

  return seq;
}
    
void
VolVis::LinearAChanged( clString a )
{
  a.get_double( newLinearA );
  msgs.send( NEWLINEARATTENUATION );
}

void
VolVis::TransferMapChanged()
{
  msgs.send( NEWTRANSFERMAP );
}

void
VolVis::StepSizeChanged( clString a )
{
  int r;
  a.get_int(r);
  stepcountChanged = r;
  
  msgs.send( NEWSTEP );
}


void
VolVis::ViewChanged( clString /* a */)
{
  // if connected to a Salmon module, retrieve Salmon geometry info
  // and store it in a LevoyS type structure
  // otherwise, use the Levoy structure
  if ( salmonData.get() )
    {
      GeometryData *v = ogeom->getData(0, GEOM_VIEW);
      if ( v != NULL )
	{
	  newview = *(v->view);
#if 0	  
	  ExtendedView joy( *(v->view), myview.xres(), myview.yres(), myview.bg() );
	  imagelock.lock();
	  myview = joy;
	  imagelock.unlock();
#endif	  
	}
      else
	cerr << "Invalid view from Salmon\n";
    }
  else
    {
      reset_vars();
      newview = (View) iEView.get();
#if 0      
      reset_vars();
      imagelock.lock();
      myview = iEView.get();
      imagelock.unlock();
#endif      
    }

  msgs.send( NEWVIEW );
}

void
VolVis::RenderLoop( )
{
  int msg;
  int i;
  done = 1;
  int row, col;
  unsigned int seq, start_seq;

  WallClockTimer watch;
  watch.start();
  
  int NewRaster = 0;
  int NewView   = 0;
  int NewScalarField = 0;
  int NewTransferMap = 0;
  int InvalidScalarField = 1;
  int NewBackground = 0;
  int NewColorMap = 0;

  double oneover255 = 1.0 / 255;
  Vector rayToTrace;
  Point beg;

  double r, g, b;
    
  col = 1;
  row = 0;
  xmiddle = ymiddle = 50;
  seq = 0x0001;

  for(;;)
    {
      msg = -1;
      
      if ( done )
	msg = msgs.receive();
      else
	msgs.try_receive( msg );
      
      while ( msg != -1 )
	{
	  /* do the swich here, update variables */
	  switch( msg )
	    {
	    case NEWSCALARFIELD:
	      {
		// cerr <<"Got SF\n";
		InvalidScalarField = 0;
		NewScalarField = 1;
		break;
	      }
	    case NEWVIEW:
	      {
		// cerr <<"Got view\n";
		NewView = 1;
		break;
	      }
	    case NEWTRANSFERMAP:
	      {
		// cerr <<"Got trans\n";
		NewTransferMap = 1;
		break;
	      }
	    case NEWRASTER:
	      {
		NewRaster = 1;
		//cerr <<"Got rast " << myview.xres() << " " << myview.yres() << endl;
		break;
	      }
	    case NEWSTEP:
	      {
		//cerr <<"Got new step size\n";
		rayStep /= stepcount;
		rayStep *= stepcountChanged;
		stepcount = stepcountChanged;
		break;
	      }
	    case NEWINVALIDSCALARFIELD:
	      {
		InvalidScalarField = 1;
		break;
	      }
	      
	    case NEWBACKGROUND:
	      {
		NewBackground = 1;
		break;
	      }

	    case NEWLINEARATTENUATION:
	      {
		LinAtten = newLinearA;
		break;
	      }
	    case NEWColorMap:
	      {
		NewColorMap = 1;
		break;
	      }
	    }
	  
	  // memorize the starting sequence but make sure it's
	  // within the bounds for that mask (right now i'm doing
	  // this for mask of width 14

	  start_seq = seq;
	  // set the global variables
	  global_start_seq = start_seq;
	  global_seq = seq;
	  
	  done = 0;
	  watch.clear();
	  
	  msg = -1;
	  msgs.try_receive( msg );
	}

      if ( NewTransferMap )
	{
	  UTArrays();

	  CreateTransferTables();
	  NewTransferMap = 0;
	  
	  if ( whiteFlag )
	    CastRay = &BasicVolRender;
	  else
	    CastRay = &ColorVolRender;
	}

      if ( NewColorMap )
	{
	  SVR = cmap->rawRed;
	  SVG = cmap->rawGreen;
	  SVB = cmap->rawBlue;
	  SVOpacity = cmap->rawAlpha;

	  cerr << "getting a new ColorMap\n";
	  CastRay = &ColorVolRender;
	  NewColorMap = 0;
	}
      
      if ( NewScalarField )
	{
	  homeSFRGrid = data;
	  homeSFRGrid->get_minmax( min, max );
	  minSV.set( min );
	  maxSV.set( max );

	  // reset the bbox, the voxel counts, and the diagonal
	  homeSFRGrid->get_bounds( bmin, bmax );
	  box.reset();
	  box.extend( bmin );
	  box.extend( bmax );
//	  cerr << "\n\nPREPARING FOR INTERSECTION WITH " << myview.eyep() << endl;
	  box.PrepareIntersect( myview.eyep() );
	  
	  diagonal = bmax - bmin;
	  diagx = diagonal.x();
	  diagy = diagonal.y();
	  diagz = diagonal.z();

	  nx = homeSFRGrid->nx;
	  ny = homeSFRGrid->ny;
	  nz = homeSFRGrid->nz;

	  rayStep = DetermineRayStepSize( stepcount );
	  at();
	  CreateTransferTables();
	  at();
	  NewScalarField = 0;
	}
      
      if ( NewBackground )
	{
	  myview.bg( newbg * oneover255 );
	  mbgr = myview.bg().r();
	  mbgg = myview.bg().g();
	  mbgb = myview.bg().b();
	  
	  NewBackground = 0;
	}
	
      if ( NewRaster )
	{
	  xres = xresChanged;
	  yres = xresChanged;
	  myview.xres( xres );
	  myview.yres( yres );


	  // clear the GLwindow to background color
	  
//	  glClearColor( myview.bg().r(), myview.bg().g(), myview.bg().b(), 0 );
  
//	  glClear(GL_COLOR_BUFFER_BIT);


//	  cerr << "got a new raster: " << xres << "  " << yres << endl;
	  
	  xmiddle = xres / 2;
	  ymiddle = yres / 2;
  
	  seq = FindMask( seq );
	  start_seq = seq;
	  
	  CalculateRayIncrements();

	  NewRaster = 0;

	  //OPENGL	  Image.newsize( xres, yres );
	}

      if ( NewView )
	{
	  ExtendedView joy( newview, xres, xres, myview.bg() );
	  myview = joy;
	  
	  eye = myview.eyep();
	  homeRay = myview.lookat() - eye;
	  homeRay.normalize();
	  CalculateRayIncrements();
	  
	  box.PrepareIntersect( myview.eyep() );

	  Color bg(myview.bg());
	  Image.initialize( bg );
  
	  NewView = 0;
	}

      // if rdata valid, render a bunch of rays
      if ( ! InvalidScalarField )
	{
	  if ( iProc.get() )
	    {
	      procCount = Task::nprocessors();
	      Task::multiprocess(procCount, do_parallel, this);

	      if ( done )
		cerr << "Parallel reports " << watch.time();

	      global_seq = new_seq;

	      redraw_all();
	    }
	  else
	    {
	      imagelock.lock();
	      for ( i = 0; i < 5000; i++ )
		{
		  row = seq / xres;
		  col = seq % xres;
		  
		  if ( row < yres )
		    {
		      // CAST A RAY

		      rayToTrace = homeRay;
		      rayToTrace += rayIncrementU * ( col - xmiddle );
		      rayToTrace += rayIncrementV * ( row - ymiddle );
		      rayToTrace.normalize();

		      // make the ray the length of the step
		      rayToTrace *= rayStep;

		      // must add the offset(to put the image in the middle)
		      
		      if ( box.Intersect( eye, rayToTrace, beg ) )
			{
			  (*CastRay)( rayToTrace.x(), rayToTrace.y(),
				  rayToTrace.z(), rayStep,
				     beg.x(),
				     beg.y(), beg.z(), SVOpacity, SVR,
				     SVG, SVB, min,
				     SVMultiplier, box.min().x(),
				     box.min().y(), box.min().z(),
				     homeSFRGrid->grid.get_dataptr(),
				     mbgr, mbgg, mbgb,
				     nx, ny, nz,
				     diagx, diagy, diagz, LinAtten,
				     &r, &g, &b );
			  
			  Image( row, col ).red   = (unsigned char)(r*255.0);
			  Image( row, col ).green = (unsigned char)(g*255.0);
			  Image( row, col ).blue  = (unsigned char)(b*255.0);
			}
		      else
			{
			  Image( row, col ).red   = (unsigned char)(mbgr*255.0);
			  Image( row, col ).green = (unsigned char)(mbgg*255.0);
			  Image( row, col ).blue  = (unsigned char)(mbgb*255.0);
			}
		    }
		  
		  // compute the next sequence element
		  if ( (ONE & seq) == ONE )
		    seq = (seq>>1) ^ ( mask );
		  else
		    seq = seq>>1;

		  if ( seq == start_seq )
		    {
		      done = 1;
		      cerr << "The timer reports: " << watch.time() << endl;
		      break;
		    }
		}
	      imagelock.unlock();

	      // STEVE : i wanted to redraw only when idle, but i failed
	      // because there was some kind of a dead lock somewhere.
	      // so, i'm just redrawing on the fly (it looks horrible,
	      // but what can i do?
//	      update_progress(0.5);
//	      TCL::execute(id+" redraw_when_idle");
	      redraw_all();
	    }
	}
      else
	{
	  cerr << "The grid is invalid\n";
	  done = 1;
	}
    }
}

int
RenderingThread::body( int )
{
  v->RenderLoop();
  return 1;
}


void
VolVis::CreateTransferTables()
{
  int i,j,k;
  int beg, end;
  int counter;

  // make sure these point to local data
  SVR = SVRsave;
  SVG = SVGsave;
  SVB = SVBsave;
  SVOpacity = SVOpacitysave;
  
  // scalar value multiplier which scales the SV appropriately
  // for array access
  SVMultiplier = ( VTABLE_SIZE - 1 ) / ( max - min );

  at();
  j = 0;
  counter = 0;
  SVOpacity[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVOpacity[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
				SVMultiplier + AssociatedVals[j][i] );
    }

  at();
  j = 1;
  counter = 0;
  SVR[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVR[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }

  at();
  j = 2;
  counter = 0;
  SVG[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVG[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }

  at();
  j = 3;
  counter = 0;
  SVB[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVB[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }

  at();
  if ( counter > VTABLE_SIZE )
    {
      cerr << "NOT GOOD AT ALL\n";
      ASSERT( counter == VTABLE_SIZE );
    }
}

/**************************************************************************
 *
 * attempts to determine the most appropriate interval which is used
 * to step through the volume.
 *
 **************************************************************************/

double
VolVis::DetermineRayStepSize ( int steps )
{
  double small[3];
  double result;
  
  // calculate a step size that is about the length of one voxel
  small[0] = ( box.max().x() - box.min().x() ) / nx;
  small[1] = ( box.max().y() - box.min().y() ) / ny;
  small[2] = ( box.max().z() - box.min().z() ) / nz;

  // set rayStep to the smallest of the step sizes
  if ( small[0] < small[1] )
    if ( small[0] < small[2] )
      result = small[0];
    else
      result = small[2];
  else if ( small[1] < small[2] )
    result = small[1];
  else
    result = small[2];

  return( steps * result );
}

void
VolVis::CalculateRayIncrements ()
{
  imagelock.lock();
  myview.get_normalized_viewplane( rayIncrementU, rayIncrementV );

  double aspect = double( myview.xres() ) / double( myview.yres() );
  double fovy=RtoD(2*Atan(aspect*Tan(DtoR(myview.fov()/2.))));
  
  double lengthY = 2 * tan( DtoR( fovy / 2 ) );

  rayIncrementV *= lengthY / myview.yres();
  rayIncrementU *= lengthY / myview.yres();
  imagelock.unlock();
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:58:17  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:30  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:03  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
