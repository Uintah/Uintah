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
 *  OLA!
 *
 * Depth cueing
 * Changing light positions / adding lights
 * adjustable min/max opacity/transparency
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>

#include <Geom/GeomOpenGL.h>
#include <Geom/Triangles.h>
#include <Geom/TCLView.h>
#include <Geom/View.h>
#include <Geom/tGrid.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <VolPack/volpack.h>

#include <iostream.h>
#include "kuswik.h"

#define VP__NUM_FIELDS            3
#define VP__NUM_SHADE_FIELDS      1
#define VP__NUM_CLASSIFY_FIELDS   2

#define VP__NORMAL_FIELD          0
#define VP__NORMAL_MAX            VP_NORM_MAX
#define VP__SCALAR_FIELD          1
#define VP__SCALAR_MAX            255
#define VP__GRAD_FIELD            2
#define VP__GRAD_MAX              VP_GRAD_MAX

#define VP__COLOR_CHANNELS        3
#define VP__NUM_MATERIALS         1

typedef unsigned char Scalar;
typedef unsigned short Normal;
typedef unsigned char Gradient;
typedef struct
{
  Normal normal;
  Scalar scalar;
  Gradient gradient;
} Voxel;



#define VV_VIEW_PORT_SIZE   600

#define NONE                  0
#define RASTER_CHANGED        1
#define VIEW_CHANGED          2
#define CMETHOD_CHANGED       4
#define TRANSFER_MAP_CHANGED  8
#define SCALAR_FIELD_CHANGED 16

#define WITH_OCTREE           1

const View HOME_VIEW
(Point(0.5,0.5,1), Point(0.5, 0.5, 0.5), Vector(0, 1, 0), 60 );

const Color RED ( 1, 0, 0);
const Color GREEN ( 0, 1, 0);
const Color BLUE ( 0, 0, 1);
const Color YELLOW ( 1, 1, 0);
const Color PURPLE ( 1, 0, 1);
const Color CYAN ( 0, 1, 1);
const Color CUSTOM ( 0.5, 0.3, 0.7);

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);


/**************************************************************
 *
 * This is the base class for the Volume visualization module.
 *
 **************************************************************/

class BB : public Module {

  GeometryOPort* aGeometry;        // Output port to salmon

  ScalarFieldIPort *aScalarField;  // Scalar field
  ScalarFieldHandle aScalarFieldH; // Scalar field handle
  ScalarFieldRG *aRGData;          // Pointer to regularly gridded
                                   // scalar field data
  
  ColorMapIPort* aColorMap;        // Transfer map
  ColorMapHandle aColorMapH;       // Transfer map handle

  int aScalarFieldGeneration;      // Generation number for sf
  int aColorMapGeneration;         // Generation number for transfer map

  
  TCLView a_tcl_View;              // view info from the UI
  View aView;                      // view info

  TCLint a_tcl_Raster;             // raster size from the UI
  int aRaster;                     // raster size
  
  TCLint a_tcl_classify_method;    // determines whether the algorithm
                                   // uses the octree (WITH_OCTREE) or not
  
  int flag;                        // Determines the work to be done by execute
                                   // This variable is based on UI events

  
  Window win;                      // Window used by OpenGL/TCL
  Display* dpy;                    // Display used by OpenGL/TCL

  
  unsigned char *aImage;           // Final image

  vpContext *vpc;                  // Rendering context used by VolPack
  

  GeomTrianglesPC* tris;           // Debug: list of bbox triangles
  int aGeometryID,texGeomID;       // ID's for bbox and texture sent to Salmon

  Vector p1,p2,p3,p4;              // Points bounding the bbox projection
                                   // onto the viewing plane

  // OLA! 
  void ProjectPoint( Vector v );   // Projects a single point on the view plane
  Point eye;                     
  Vector z, up, x;
  double minX, minY, maxX, maxY;
  
public:

  /* Standard procedures */
  BB(const clString& id);          // Constructor
  BB(const BB&, int deep);         // Copy Constructor (not finished)
  virtual ~BB();                   // Destructor
  virtual Module* clone(int deep); // Cloning function
  virtual void execute();          // Execute the module
  void tcl_command( TCLArgs&, void * ); // Catches calls from the TCL interface

  void redraw_all();               // Redraw the image
  int makeCurrent();               // Create an OpenGL window

  void Setup();                    // Sets up the constant VolPack variables
  
  void PreProcess();               // Stores important information from a new
                                   // scalar grid into VolPack's vpc
  
  void Render();                   // Renders volume given a change in view
  void RenderBRaster();            // Renders volume based on a raster size
                                   // change
  
  void Classify();                 // Classifies the data
  void UseOctree();                // Creates the octree
  void DestroyOctree();            // Destroys the octree
};



/**************************************************************
 *
 * Create a new instance of the module.
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
  a_tcl_Raster("raster", id, this),
  a_tcl_classify_method("cmethod", id, this),
  aGeometryID(0),texGeomID(0)
{
  // Create scalar field input port
  aScalarField = scinew
    ScalarFieldIPort(this, "HOMESFRGRID", ScalarFieldIPort::Atomic);
  add_iport(aScalarField);

  // Create transfer map input port
  aColorMap = scinew
    ColorMapIPort( this, "Transfer Map", ColorMapIPort::Atomic);
  add_iport(aColorMap);

  // Create salmon output port
  aGeometry = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(aGeometry);

  // Default view info
  a_tcl_View.set( HOME_VIEW );

  // Default raster size
  a_tcl_Raster.set( 100 );
  aRaster = 100;

  // Default classification method : use octree
  a_tcl_classify_method.set( 0 );

  // Allocate memory for the final image array
  aImage = new unsigned char[VV_VIEW_PORT_SIZE*VV_VIEW_PORT_SIZE*3];
  ASSERT( aImage != NULL );

  // No changes necessary for recomputation of the final image exist
  flag = NONE;

  // Set VolPack variables that will not change during the visualization
  Setup();
}

/**************************************************************
 *
 * Another constructor
 *
 **************************************************************/

BB::BB(const BB& copy, int deep)
: Module(copy, deep),
  a_tcl_View("eview", id, this),
  a_tcl_Raster("raster", id, this),
  a_tcl_classify_method("cmethod", id, this)
{
  NOT_FINISHED("BB::BB");
}

/**************************************************************
 *
 * Destructor
 * OLA! perhaps some data should be freed!
 *
 **************************************************************/

BB::~BB()
{
}

/**************************************************************
 *
 * Clones the module
 *
 **************************************************************/

Module*
BB::clone(int deep)
{
  return scinew BB(*this, deep);
}


/**************************************************************
 *
 * Initializes OpenGL variables
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

  // set the viewport size to the max image size
  glViewport(0, 0, VV_VIEW_PORT_SIZE, VV_VIEW_PORT_SIZE);

  // set the projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, VV_VIEW_PORT_SIZE-1, VV_VIEW_PORT_SIZE-1, 0, -1, 1);

  // set the modelview matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  return 1;
}


/**************************************************************
 *
 * Redraws the image stored in aImage (does not recalculate the image)
 *
 **************************************************************/

void
BB::redraw_all()
{
  // initialize OpenGL variables
  if (!makeCurrent()) return;

  // provide double buffering
  glDrawBuffer(GL_BACK);
  
  // clear the screen to black
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
  
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // scale the image to the window size => this should be
  // glPixelZoom(zoom,zoom); but that was too slow on singular
  double zoom = VV_VIEW_PORT_SIZE * 1.0 / aRaster;
  if ( zoom >= 2 )
    glPixelZoom(2,2);
  else
    glPixelZoom(1,1);
  
  glRasterPos2i( 0, VV_VIEW_PORT_SIZE-1 );

  // draw the image
  glDrawPixels( aRaster, aRaster, GL_RGB, GL_UNSIGNED_BYTE, aImage );
  glXSwapBuffers(dpy,win);
  
  glXMakeCurrent(dpy, None, NULL);

  // unlock TCLTask which was locked in makeCurrent
  TCLTask::unlock();
}	    


/**************************************************************
 *
 * Catches calls from the UI code.  Sets the flag variable which
 * tells the user what changes image recalculation must deal with.
 * Calls want_to_execute which causes execute to be called by the
 * scheduler.
 *
 **************************************************************/

void
BB::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "redraw_all") // UI requests an OpenGL screen redraw
    redraw_all();
  else if (args[1] == "view_changed") // view change notification
    {
      if ( ! (flag & VIEW_CHANGED) )
	{
	  flag |= VIEW_CHANGED;
	  want_to_execute();
	}
    }
  else if (args[1] == "raster_changed") // raster changed
    {
      if ( ! flag ) // any changes that are in the queue will appropriately
	{           // deal with a raster change
	  flag |= RASTER_CHANGED;
	  want_to_execute();
	}
    }
  else if (args[1] == "classify_changed") // classify method changed
    {
      if ( !(flag & CMETHOD_CHANGED) )
	{
	  flag |= CMETHOD_CHANGED;
	  want_to_execute();
	}
    }
  else // some other tcl request took place
    Module::tcl_command(args, userdata);
}



/**************************************************************
 *
 * Execute can happen when:
 *  - i tell the module to execute
 *  - the scalar field has changed
 *  - the transfer map has changed
 *  - the view has changed
 *  - the raster has changed
 *
 * Variable "flag" tells execute which one(s) of the above changes
 * took place.  Based on that info, execute recalculates the final
 * image.
 *
 * Since this module must be connected to the ScalarFieldReader
 * and GenTransferMap modules, execute reassures this is the case.
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

  // reset variables provided by the UI (basically, update tcl info
  // to the recent changes made in the UI)
  reset_vars();

  // set flag if a new scalar field or colormap has been read.
  // the generation numbers will be different if either the
  // ScalarFieldReader or the GenTransferMap  executed
  if ( aScalarFieldGeneration != aScalarFieldH->generation )
    flag |= SCALAR_FIELD_CHANGED;
  else if ( aColorMapGeneration != aColorMapH->generation )
    flag |= TRANSFER_MAP_CHANGED;
  else if ( flag == NONE ) // tell me if execute takes place for no reason
    cerr << "REEXECUTING\n";
    
  while ( flag != NONE ) // recalculate the image based on all the requests
    {
      if ( flag & CMETHOD_CHANGED ) // with octree vs. no octree flag
	{
	  flag &= ~CMETHOD_CHANGED;
	  
	  if ( a_tcl_classify_method.get() == WITH_OCTREE )
	    UseOctree();            // calculate the octree
	  else
	    DestroyOctree();        // destroy the octree
	}
      else if ( flag & SCALAR_FIELD_CHANGED ) // SCALAR FIELD CHANGED
	{
	  // memorize the raster size
	  aRaster = a_tcl_Raster.get();

	  // OLA! get rid of if using Salmon stuff
	  // get viewing info from the UI
	  aView = a_tcl_View.get();

	  // recalculate the image from scratch
	  flag = NONE;
	  
	  // save the generation numbers
	  aScalarFieldGeneration = aScalarFieldH->generation;
	  aColorMapGeneration = aColorMapH->generation;

	  // check if the grid is valid (if it's regular)
	  aRGData = aScalarFieldH->getRG();
	  
	  if ( aRGData == NULL )
	    {
	      // check if it is an irregular grid
	      ScalarFieldUG *UGData = aScalarFieldH->getUG();
	      
	      if ( UGData != NULL )
		{
		  cout << "This is an irregular grid\n";
		}
	      else
		{
		  cerr << "This is an invalid grid\n";
		  
		  // Invalid data -- clear the screen
		  int i, j;
		  for (i=0;i<VV_VIEW_PORT_SIZE*VV_VIEW_PORT_SIZE;i++)
		      aImage[i] = 0;
		}
	    }
	  else
	    {
	      cout << "This is a regular grid\n";

	      // preprocess the data and render
	      PreProcess();

	      // get the view information from Salmon
	      GeometryData *v = aGeometry->getData(0, GEOM_VIEW);
	      aView = *(v->view);

	      cerr << "The view info is! " << aView.eyep() << endl;
	      cerr << "The view info is! " << aView.lookat() << endl;
	      cerr << "The view info is! " << aView.up() << endl;
	      
	      Render();

	    }
	}
      else if ( flag & TRANSFER_MAP_CHANGED )
	{
	  /*** Transfer map has changed ***/
	  
	  // memorize the raster size
	  aRaster = a_tcl_Raster.get();

	  flag &= ~TRANSFER_MAP_CHANGED;
	  flag &= ~RASTER_CHANGED;
	  Classify();
	  RenderBRaster();
	}
      else if ( flag & VIEW_CHANGED )
	{
	  // OLA! get rid of if using Salmon stuff
	  aView = a_tcl_View.get();

	  // memorize the raster size
	  aRaster = a_tcl_Raster.get();

	  flag &= ~VIEW_CHANGED;
	  flag &= ~RASTER_CHANGED;
	  
	  // get the view information from Salmon
	  GeometryData *v = aGeometry->getData(0, GEOM_VIEW);
	  aView = *(v->view);
	  
	  Render();
	}
      else if ( flag & RASTER_CHANGED )
	{
  // memorize the raster size
  aRaster = a_tcl_Raster.get();

	  flag &= ~RASTER_CHANGED;
	  RenderBRaster();
	}
      else
	cerr << "Re-executing\n";
    }
}



/**************************************************************
 *
 *
 **************************************************************/

void
BB::Setup()
{
  int vpres;
  Voxel *dummy_voxel = new Voxel;

  cerr << "Setup\n";

  
  /****************************************************************
   * Rendering Context
   ****************************************************************/

  // Create a new context
  vpc = vpCreateContext();
  if ( vpc == NULL ) cerr << "context is NULL\n";

  /****************************************************************
   * Voxel description
   ****************************************************************/

  // Declare the size of the voxel and the number of fields it contains
  vpres = vpSetVoxelSize ( vpc, sizeof(Voxel), VP__NUM_FIELDS,
			  VP__NUM_SHADE_FIELDS, VP__NUM_CLASSIFY_FIELDS );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

  // Declare the size and position of each field within the voxel
  // Normal
  vpres = vpSetVoxelField( vpc, VP__NORMAL_FIELD, sizeof(dummy_voxel->normal),
			  vpFieldOffset(dummy_voxel, normal), VP__NORMAL_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField1" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // Gradient
  vpres = vpSetVoxelField(vpc, VP__GRAD_FIELD, sizeof(dummy_voxel->gradient),
			  vpFieldOffset(dummy_voxel, gradient), VP__GRAD_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField3" << vpGetErrorString(vpGetError(vpc)) << "\n";

  /****************************************************************
   * Additional settings
   ****************************************************************/
  
  // voxels of <=0.05 opacity are transparent
  vpres = vpSetd(vpc, VP_MIN_VOXEL_OPACITY, 0.05);
  if ( vpres != VP_OK )
    cerr << "vpSetd " << vpGetErrorString(vpGetError(vpc)) << endl;

  // set the threshhold to be 95%
  vpres = vpSetd(vpc, VP_MAX_RAY_OPACITY, 0.95);
  if ( vpres != VP_OK ) cerr << "vpSetd(Rendering) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // set pre-multiplication of matrices
  vpres = vpSeti(vpc, VP_CONCAT_MODE, VP_CONCAT_LEFT );
  if ( vpres != VP_OK ) cerr << "vpSeti " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  //
  // additional gradient based classification
  //

  float *gradTable = new float[VP_GRAD_MAX+1];
  
  vpres = vpSetClassifierTable(vpc, 1, VP__GRAD_FIELD, gradTable,
			       (VP_GRAD_MAX+1)*sizeof(float));
  if ( vpres != VP_OK )
    cerr <<"vpSetClassifierTable "<<vpGetErrorString(vpGetError(vpc)) << endl;

  int GradientRampPoints = 4;
  int GradientRampX[] = {    0,   5,  20, 221};
  float GradientRampY[] = {0.0, 0.0, 1.0, 1.0};
  vpres = vpRamp(gradTable, sizeof(float), GradientRampPoints,
		 GradientRampX, GradientRampY);
  if ( vpres != VP_OK )
    cerr <<"vpRamp "<<vpres <<" "<< vpGetErrorString(vpGetError(vpc)) << endl;


}

void
BB::PreProcess()
{
  cerr << "Welcome in Preprcess\n";
  
  double smin, smax;
  int densitySize, volumeSize;
  int vpres;
  Voxel *dummy_voxel = new Voxel;

  // find the min and max scalar field values
  aRGData->get_minmax(smin,smax);

  /****************************************************************
   * Volumes
   ****************************************************************/

  // Set volume dimensions
  vpres = vpSetVolumeSize( vpc, aRGData->nx, aRGData->ny, aRGData->nz );
  if ( vpres != VP_OK )
    cerr << "vpSetVolumeSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

  // Scalar
  vpres = vpSetVoxelField(vpc, VP__SCALAR_FIELD, sizeof(dummy_voxel->scalar),
			  vpFieldOffset(dummy_voxel, scalar), smax );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField2" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // determine array sizes
  densitySize = aRGData->nx * aRGData->ny * aRGData->nz;
  volumeSize = densitySize * sizeof(Voxel);
  
  // store the scalar values in a one dimensional array
  unsigned char *density = new unsigned char[densitySize];
  (aRGData->grid).get_onedim_byte( density );

  // allocate memory for the volume
  Voxel * volume = new Voxel[densitySize];

  // check if memory was allocated
  if (density == NULL || volume == NULL) cerr << "out of memory\n";

  vpres = vpSetRawVoxels(vpc, volume, volumeSize, sizeof(Voxel),
			 aRGData->nx * sizeof(Voxel),
			 aRGData->nx * aRGData->ny * sizeof(Voxel));
  if ( vpres != VP_OK ) cerr << "vpSetRawVoxels" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  vpres = vpVolumeNormals(vpc, density, densitySize, VP__SCALAR_FIELD,
			  VP__GRAD_FIELD, VP__NORMAL_FIELD);
  if ( vpres != VP_OK ) cerr << "vpVolumeNormals " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  Classify();
}


void
BB::UseOctree()
{
  int vpres;

  cerr <<"using the octree\n";
  
  vpres = vpMinMaxOctreeThreshold( vpc, 0, 4 );
  if ( vpres != VP_OK )
    cerr << "threshold " << vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpMinMaxOctreeThreshold( vpc, 1, 4 );
  if ( vpres != VP_OK )
    cerr << "threshold(2) " << vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpCreateMinMaxOctree( vpc, 0, 4 );
  if ( vpres != VP_OK )
    cerr << "CreateOctree " << vpGetErrorString(vpGetError(vpc)) << endl;
}


void
BB::DestroyOctree()
{
  cerr <<"Destroyed\n";
  if ( vpDestroyMinMaxOctree(vpc) != VP_OK )
    cerr << "Destroying! "<< vpGetErrorString(vpGetError(vpc)) << endl;
}


void
BB::Classify()
{
  cerr << "Welcome in Classify\n";
  
  int vpres;
  double smin, smax;
  
  // find the min and max scalar field values
  aRGData->get_minmax(smin,smax);
  
  /****************************************************************
   * Classification
   ****************************************************************/

  // allocate the classification table
  float *scalarTable = new float[(int)smax+1];

  // find the difference in min and max scalar value
  double diff = smax - smin;
  
  Array1<int> xl;  // = *(aColorMapH->rawRampAlphaT);
  Array1<float> sv = *(aColorMapH->rawRampAlpha);

  // convert percentages to scalar values
  int j;
  for(j=0;j<sv.size();j++)
    xl.add( (int)( (*(aColorMapH->rawRampAlphaT))[j] * diff) );

  // place the classification table in the rendering context
  vpres = vpSetClassifierTable(vpc, 0, VP__SCALAR_FIELD, scalarTable,
			       (smax+1)*sizeof(float));
  if ( vpres != VP_OK )
    cerr <<"vpSetClassifierTable "<<vpGetErrorString(vpGetError(vpc)) << endl;

  // create the classification by specifying a ramp
  vpres = vpRamp( scalarTable, sizeof(float), xl.size(), xl.get_objs(),
		 sv.get_objs());
  if ( vpres != VP_OK )
    cerr <<"vpRamp "<<vpres <<" "<< vpGetErrorString(vpGetError(vpc)) << endl;

  if ( a_tcl_classify_method.get() == WITH_OCTREE )
    UseOctree();

  /****************************************************************
   * Classify volume
   ****************************************************************/

  vpres = vpClassifyVolume(vpc);
  if ( vpres != VP_OK ) cerr << "vpClassifyVolume " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
}



void
BB::Render()
{
  cerr << "Welcome to Render\n" << aView.up() <<endl;
  int vpres;
  /****************************************************************
   * View Transformations
   ****************************************************************/

  Point bboxmin, bboxmax;
  // find bbox min and max corners
  aRGData->get_bounds( bboxmin, bboxmax );
  
  // set up the modeling matrix
  vpres = vpCurrentMatrix(vpc, VP_MODEL);
  if ( vpres != VP_OK ) cerr << "vpCurrentMatrix(1) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpIdentityMatrix(vpc);
  if ( vpres != VP_OK ) cerr << "vpIdentity(1) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpTranslate(vpc, bboxmin.x()-0.5, bboxmin.y()-0.5, bboxmin.z()-0.5);
  if ( vpres != VP_OK ) cerr << "vpTranslate(1b) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // create the view transformation matrix
  eye = aView.eyep();
  z = aView.lookat() - eye;
  z.normalize();
  up = aView.up() - z * Dot(z,aView.up());
  // backwards  up = z * Dot(z,aView.up()) - aView.up();
  up.normalize();
  x = Cross(z,up);
  x.normalize();

  vpMatrix4 m;
  m[0][0] = x.x();
  m[0][1] = x.y();
  m[0][2] = x.z();
  m[0][3] = 0.;
  m[1][0] = up.x();
  m[1][1] = up.y();
  m[1][2] = up.z();
  m[1][3] = 0.;
  m[2][0] = z.x();
  m[2][1] = z.y();
  m[2][2] = z.z();
  m[2][3] = 0.;
  m[3][0] = 0.;
  m[3][1] = 0.;
  m[3][2] = 0.;
  m[3][3] = 1.;

  // set up the view matrix
  vpres = vpCurrentMatrix(vpc, VP_VIEW);
  if ( vpres != VP_OK ) cerr << "vpCurrent(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpIdentityMatrix(vpc);
  if ( vpres != VP_OK ) cerr << "vpIdentity(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpTranslate(vpc, eye.x(), eye.y(), eye.z());
  if ( vpres != VP_OK ) cerr << "vpTranslate(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpMultMatrix(vpc, m);
  if ( vpres != VP_OK ) cerr << "vpMult(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // calculate the bbox projection onto the viewing plane

  double dx,dy, maxX, minX, maxY, minY;

  Vector one = bboxmin - eye;
  Vector two = bboxmax - eye;

  cerr << "x : " << x << endl;
  cerr << "up : " << up << endl;
  cerr << one << endl;
  cerr << two << endl;

  minX = maxX = one.x() * x.x() + one.y() * x.y() + one.z() * x.z();
  dx = one.x() * x.x() + one.y() * x.y() + two.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = one.x() * x.x() + two.y() * x.y() + one.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = one.x() * x.x() + two.y() * x.y() + two.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = two.x() * x.x() + one.y() * x.y() + one.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = two.x() * x.x() + one.y() * x.y() + two.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = two.x() * x.x() + two.y() * x.y() + one.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
  dx = two.x() * x.x() + two.y() * x.y() + two.z() * x.z();
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;

  minY = maxY = one.x() * up.x() + one.y() * up.y() + one.z() * up.z();
  dy = one.x() * up.x() + one.y() * up.y() + two.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = one.x() * up.x() + two.y() * up.y() + one.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = one.x() * up.x() + two.y() * up.y() + two.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = two.x() * up.x() + one.y() * up.y() + one.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = two.x() * up.x() + one.y() * up.y() + two.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = two.x() * up.x() + two.y() * up.y() + one.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  dy = two.x() * up.x() + two.y() * up.y() + one.z() * up.z();
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;

  cerr << "The numbers are: ( " << minX << ", " << minY << " ) ( " << maxX
    << ", " << maxY << " )\n";

  this->minX = this->minY = this->maxX = this->maxY = 0;
  ProjectPoint( one );
  ProjectPoint( Vector( one.x(), one.y(), two.z() ) );
  ProjectPoint( Vector( one.x(), two.y(), one.z() ) );
  ProjectPoint( Vector( one.x(), two.y(), two.z() ) );
  ProjectPoint( Vector( two.x(), one.y(), one.z() ) );
  ProjectPoint( Vector( two.x(), one.y(), two.z() ) );
  ProjectPoint( Vector( two.x(), two.y(), one.z() ) );
  ProjectPoint( two );
  
  cerr << "The numbers are: ( " << this->minX << ", " <<
    this->minY << " ) ( " << this->maxX
    << ", " << this->maxY << " )\n";

  maxX = this->maxX;
  maxY = this->maxY;
  minX = this->minX;
  minY = this->minY;

  Vector VolumeCenter ( (bboxmax - bboxmin) * 0.5 + bboxmin.vector() );

  p1 = Vector( VolumeCenter + x * minX + up * minY);
  p2 = Vector( VolumeCenter + x * maxX + up * minY);
  p3 = Vector( VolumeCenter + x * minX + up * maxY);
  p4 = Vector( VolumeCenter + x * maxX + up * maxY);

  cerr << "P1 " << p1 << endl;
  cerr << "P2 " << p2 << endl;
  cerr << "P3 " << p3 << endl;
  cerr << "P4 " << p4 << endl;

  // set up the projection matrix
  vpres = vpCurrentMatrix(vpc, VP_PROJECT);
  if ( vpres != VP_OK ) cerr << "vpCurrent(3) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpIdentityMatrix(vpc);
  if ( vpres != VP_OK ) cerr << "vpIdentity(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpWindow(vpc, VP_PARALLEL, minX, maxX, minY, maxY, -1, 1);
  if ( vpres != VP_OK ) cerr << "vpWindow(3) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  /****************************************************************
   * Give Salmon the bbox and the rectangle coordinates of the image
   ****************************************************************/

  tris = scinew GeomTrianglesPC;
  tris->add( bboxmin, RED,
	    Point(bboxmin.x(), bboxmin.y(), bboxmax.z()), BLUE,
	    Point(bboxmin.x(), bboxmax.y(), bboxmin.z()), PURPLE );

  tris->add( bboxmin, RED,
	    Point(bboxmin.x(), bboxmax.y(), bboxmin.z()), PURPLE,
	    Point(bboxmax.x(), bboxmin.y(), bboxmin.z()), YELLOW );
  
  tris->add( bboxmin, RED,
	    Point(bboxmin.x(), bboxmin.y(), bboxmax.z()), BLUE,
	    Point(bboxmax.x(), bboxmin.y(), bboxmin.z()), YELLOW );

  tris->add( bboxmax, GREEN,
	    Point(bboxmin.x(), bboxmax.y(), bboxmax.z()), CYAN,
	    Point(bboxmax.x(), bboxmin.y(), bboxmax.z()), WHITE );
#if 0
  tris->add( p1.point(), CUSTOM, p2.point(), CUSTOM, p3.point(), CUSTOM );
  tris->add( p2.point(), CUSTOM, p3.point(), CUSTOM, p4.point(), CUSTOM );
#endif
  if ( aGeometryID )
    aGeometry->delObj(aGeometryID);
  aGeometryID = aGeometry->addObj( tris, "Bounding box and plane" );
  
  /****************************************************************
   * Shading and lighting
   ****************************************************************/

  // allocate the shade table
  int shadeTableSize = (VP_NORM_MAX+1)*VP__COLOR_CHANNELS*VP__NUM_MATERIALS;
  float *shadeTable = new float[shadeTableSize];

  // place the classification table in the rendering context
  vpres = vpSetLookupShader(vpc, VP__COLOR_CHANNELS, VP__NUM_MATERIALS,
			    VP__NORMAL_FIELD, shadeTable,
			    shadeTableSize*sizeof(float), 0, NULL, 0);
  if ( vpres != VP_OK )
    cerr << "SetLookup " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_AMBIENT, VP_BOTH_SIDES,
			0.18, 0.18, 0.18);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(a) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_DIFFUSE, VP_BOTH_SIDES,
			0.35, 0.35, 0.35);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(b) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_SPECULAR, VP_BOTH_SIDES,
			0.39, 0.39, 0.39);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(c) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_SHINYNESS, VP_BOTH_SIDES,
			10.0,0.0,0.0);
  if ( vpres != VP_OK )
    cerr << "shine " << vpGetErrorString(vpGetError(vpc)) << endl;

  //  Vector lookdir(v.lookat()-v.eyep());
  Vector lookdir(aView.eyep()-aView.lookat());
  vpres = vpSetLight(vpc, VP_LIGHT0, VP_DIRECTION,
		     lookdir.x(), lookdir.y(), lookdir.z());
  if ( vpres != VP_OK )
    cerr << "setlight " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetLight(vpc, VP_LIGHT0, VP_COLOR, 1.0, 1.0, 1.0);
  if ( vpres != VP_OK )
    cerr << "setlightcolor " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpEnable(vpc, VP_LIGHT0, 1);
  if ( vpres != VP_OK )
    cerr << "enable " << vpGetErrorString(vpGetError(vpc)) << endl;

  vpEnable( vpc, VP_LIGHT_BOTH_SIDES, 1 );
  if ( vpres != VP_OK )
    cerr << "enable(2) " << vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpShadeTable(vpc);
  if ( vpres != VP_OK )
    cerr << "shadetable " << vpGetErrorString(vpGetError(vpc)) << endl;
  
#if 0  
  /****************************************************************
   * Depth cueing
   ****************************************************************/

  vpSetDepthCueing(vpc, 0.8, 0.8);
  vpEnable(vpc, VP_DEPTH_CUE, 1);
#endif

  RenderBRaster();

}

void
BB::RenderBRaster()
{
  cerr << "Welcome to RenderBRaster\n";
  int vpres;
  
  /****************************************************************
   * Images
   ****************************************************************/

  vpres = vpSetImage(vpc, aImage, aRaster, aRaster, aRaster * 3, VP_RGB);
  if ( vpres != VP_OK ) cerr << "vpSetImage " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  /****************************************************************
   * Rendering
   ****************************************************************/

  // render the volume
  vpres = vpRenderClassifiedVolume(vpc);
  
  if ( vpres != VP_OK ) cerr << "vpRenderClassified " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // now create the textured plane...

  if (texGeomID)
    aGeometry->delObj(texGeomID);

  TexGeomGrid *tgrid = scinew TexGeomGrid(aRaster,aRaster,p1.point()
					  ,p2-p1,p3-p1);

  //  tgrid->set(aImage,VV_VIEW_PORT_SIZE);

  texGeomID = aGeometry->addObj(tgrid,"Texture Plane");
  
  aGeometry->flushViews();

  // redraw the image
  redraw_all();
      
}


void
BB::ProjectPoint( Vector v )
{
  double dx, dy;
  Vector vect = v - z * Dot(v,z);
  dx = Dot( vect, x );
  dy = Dot( vect, up );
  if ( maxY < dy ) maxY = dy;
  if ( minY > dy ) minY = dy;
  if ( maxX < dx ) maxX = dx;
  if ( minX > dx ) minX = dx;
}
  
