/*
 *  BB.cc:
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   February 1997
 *
 *  Copyright (C) 1997 SCI Group
 *
 * The following are markers in the code:
 *  PETE!
 *  OLA!
 *  STEVE!
 */

#include <Classlib/Array2.h>
#include <Geom/Color.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Timer.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/View.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColorMapPort.h>

#include <Geom/TCLView.h>
#include <Geom/View.h>

#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <Modules/Visualization/RenderRG.h>

#include <iostream.h>
#include "kuswik.h"

#define VV_REGULAR      1
#define VV_UNSTRUCTURED 2
#define VV_UNKNOWN      3

#define VV_VIEW_PORT_SIZE  600

const View HOME_VIEW
(Point(-2.0, 0.5, 0.5), Point(0.5, 0.5, 0.5), Vector(0, 1, 0), 60 );

//const View HOME_VIEW
//(Point(2.5, -3.5, 3.5), Point(0.5, 0.4, 0.3), Vector(0.3, 0.4, -0.5), 60 );

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);


/**************************************************************
 *
 * This is the base class for the Volume visualization module.
 *
 **************************************************************/

class BB : public Module {

    ScalarFieldIPort *aScalarField;     // Scalar field
    ScalarFieldHandle aScalarFieldH;    // Scalar field handle
    int aScalarFieldType;               // Scalar field type
    

    ColorMapIPort* aColorMap;           // Transfer map
    ColorMapHandle aColorMapH;          // Transfer map handle

    int aScalarFieldGeneration;         // Generation number for sf
    int aColorMapGeneration;            // Generation number for trans.map

    TCLView a_tcl_View;                 // view info from the UI

    TCLint a_tcl_Raster;                // raster size from the UI
    int aRaster;                        // raster size
    
    Window win;                         // Window used by OpenGL/TCL
    Display* dpy;                       // Display used by OpenGL/TCL

    RenderRGVolume aRGVolume;           // Class that renders a regular grid
//    Array1<CharColor> aImage;           // Final image array
    unsigned char *aImage;

    ScalarFieldRG *aRGData;    

public:

    BB(const clString& id);      // Constructor
    BB(const BB&, int deep);     // Copy Constructor (not finished)
    virtual ~BB();               // Destructor
    virtual Module* clone(int deep); // Cloning function
    virtual void execute();      // Execute the module
    void tcl_command( TCLArgs&, void * ); // Called from tcl code

    void redraw_all();           // Redraw the image
    int makeCurrent();           // Create an OpenGL window

  };



/**************************************************************
 *
 * Sets up a couple variables that all modules need to have set.
 * STEVE! what is that current_drawer variable used for???
 *
 **************************************************************/

extern "C"
{
  Module* make_BB(const clString& id)
  {
    return scinew BB(id);
  }
};

/**************************************************************
 *
 * Constructor.  Create input ports for both the scalar field
 * and the colormap.
 *
 **************************************************************/

BB::BB(const clString& id)
: Module("BB", id, Source),
  a_tcl_View("eview", id, this),
  a_tcl_Raster("raster", id, this)
{
    // Create scalar field input port
    aScalarField = scinew
      ScalarFieldIPort(this, "HOMESFRGRID", ScalarFieldIPort::Atomic);
    add_iport(aScalarField);

    // Create transfer map input port
    aColorMap = scinew
      ColorMapIPort( this, "Transfer Map", ColorMapIPort::Atomic);
    add_iport(aColorMap);

    // Default view info
    a_tcl_View.set( HOME_VIEW );

    // Default raster size
    a_tcl_Raster.set( 100 );
    aRaster = 100;

    // allocate memory for the final image array
    aImage = new unsigned char[VV_VIEW_PORT_SIZE*VV_VIEW_PORT_SIZE*3];
    ASSERT( aImage != NULL );
  }

/**************************************************************
 *
 *
 *
 **************************************************************/

BB::BB(const BB& copy, int deep)
: Module(copy, deep),
  a_tcl_View("eview", id, this),
  a_tcl_Raster("raster", id, this)
{
    NOT_FINISHED("BB::BB");
}

/**************************************************************
 *
 *
 *
 **************************************************************/

BB::~BB()
{
}

Module*
BB::clone(int deep)
{
    return scinew BB(*this, deep);
}


/**************************************************************
 *
 * this procedure initializes an open gl window.
 *
 **************************************************************/

int
BB::makeCurrent()
{
  GLXContext cx;
  Tk_Window tkwin;

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
  cx=OpenGLGetContext(the_interp, myname());

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
  glViewport(0, 0, VV_VIEW_PORT_SIZE, VV_VIEW_PORT_SIZE);
  
  glMatrixMode(GL_PROJECTION);
  
  glLoadIdentity();
  
  glOrtho(0, VV_VIEW_PORT_SIZE-1, VV_VIEW_PORT_SIZE-1, 0, -1, 1);
  
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
BB::redraw_all()
{
  WallClockTimer timer;
  timer.start();
  if (!makeCurrent()) return;
  cerr << "After makeCurrent: " << timer.time() << '\n';

  glDrawBuffer(GL_BACK);
  glClearColor( 0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT);

  // if the handle was empty, just flush the buffer (to clear the window)
  // and return.
  if (!aScalarFieldH.get_rep())
    {
      glFlush();
      glXMakeCurrent(dpy, None, NULL);
      TCLTask::unlock();
      return;
    }
  cerr << "After getRep: " << timer.time() << '\n';
  
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // scale the image to the window size
  double zoom = VV_VIEW_PORT_SIZE * 1.0 / aRaster;
  zoom=(int)zoom;
  glPixelZoom(zoom,zoom);
  
  glRasterPos2i( 0, VV_VIEW_PORT_SIZE-1 );

#if 0
  cerr << "aRaster=" << aRaster << '\n';
  cerr << "pixels[aRaster*aRaster-1]=" << (int)pixels[aRaster*aRaster-1] << '\n';
  cerr << "pid=" << getpid() << '\n';
  //glGetError(), GL_NO_ERROR, (char*)gluErrorString(errcode)
#endif
  cerr << "Before drawPixels: " << timer.time() << '\n';
  glDrawPixels( aRaster, aRaster, GL_RGB, GL_UNSIGNED_BYTE, aImage );
  cerr << "After drawPixels: " << timer.time() << '\n';

  glXSwapBuffers(dpy,win);
  cerr << "After swapBuffers: " << timer.time() << '\n';
  
  glXMakeCurrent(dpy, None, NULL);
  cerr << "After makeCurrent: " << timer.time() << '\n';
  
  TCLTask::unlock();
  cerr << "After unlock: " << timer.time() << '\n';
}	    


/**************************************************************
 *
 *
 *
 **************************************************************/

void
BB::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "redraw_all")
    redraw_all();
  else if (args[1] == "view_changed")
    want_to_execute();
  else
    Module::tcl_command(args, userdata);
}



/**************************************************************
 *
 * Detects a change in the scalar field or the colormap. Begins
 * a new calculation of the image.
 *
 **************************************************************/

void
BB::execute()
{
  // make sure that a scalar field is attached to the port
  if ( ! aScalarField->get( aScalarFieldH ) )
  { cerr << "Invalid scalar field (1)\n"; return; }
  if ( ! aScalarFieldH.get_rep() )
  { cerr << "Invalid scalar field (2)\n"; return; }
  
  // make sure that a transfer map is attached to the port
  if ( ! aColorMap->get( aColorMapH ) )
  { cerr << "Invalid transfer map\n"; return; }
  if ( ! aColorMapH.get_rep() )
  { cerr << "Invalid transfer map (2)\n"; return; }

  // reset variables provided by the UI
  reset_vars();
  
  // memorize the raster size
  aRaster = a_tcl_Raster.get();
  
  // same scalar field? -- a lot less computation if so!
  if ( aScalarFieldGeneration == aScalarFieldH->generation )
    {
      aColorMapGeneration = aColorMapH->generation;
      
      aRGVolume.Process(aRGData, aColorMapH, a_tcl_View.get(), aRaster,
			aImage);
    }
  else
  {
    aScalarFieldGeneration = aScalarFieldH->generation;
    aColorMapGeneration = aColorMapH->generation;
    
    // check if it is a regular grid
    aRGData = aScalarFieldH->getRG();
    if ( aRGData != NULL )
      {
	cout << "This is a regular grid\n";
	aScalarFieldType = VV_REGULAR;

	aRGVolume.NewScalarField( aRGData, a_tcl_View.get() );
	aRGVolume.Process(aRGData, aColorMapH, a_tcl_View.get(), aRaster,
			  aImage);
      }
    else
      {
	// check if it is an irregular grid
	ScalarFieldUG *UGData = aScalarFieldH->getUG();
	
	if ( UGData != NULL )
	  {
	    cout << "This is an irregular grid\n";
	    aScalarFieldType = VV_UNSTRUCTURED;
	  }
	else
	  {
	    cerr << "This is an invalid grid\n";
	    aScalarFieldType = VV_UNKNOWN;
	  }
      }
  }
  
  redraw_all();
}



/**************************************************************
 *
 *
 **************************************************************/

