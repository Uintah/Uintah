/*
 *
 * GeometryRenderer: Renderer dedicated to geometry rendering.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */
#include <Rendering/GeometryRenderer.h>
#include <Network/NetConversion.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <UI/GeomUI.h>
#include <UI/UserInterface.h>
#include <UI/uiHelper.h>
#include <UI/OpenGL.h>
#include <Util/Timer.h>
#include <Util/Point.h>

namespace SemotusVisum {

typedef Vector3d Vector;

Color
GeometryRenderer::geomColors[] = { Color( 255, 0, 0 ),
				   Color( 0, 255, 0 ),
				   Color( 0, 0, 255 ),
				   Color( 255, 255, 0 ),
				   Color( 255, 0, 255 ),
				   Color( 255, 128, 0 ),
				   Color( 255, 0, 128 ),
				   Color( 255, 128, 128 ),
				   Color( 128, 255, 0 ),
				   Color( 0, 255, 128 ),
				   Color( 128, 255, 128 ),
				   Color( 128, 0, 255 ),
				   Color( 0, 128, 255 ),
				   Color( 128, 128, 128 ) };

const int ON = 255;
Color 
GeometryRenderer::lightColors[] = { Color( ON, ON, ON ),
				    Color( ON, 0, 0 ),
				    Color( 0, ON, 0 ),
				    Color( 0, 0, ON ),
				    Color( ON, ON, 0 ),
				    Color( ON, 0, ON ) };

Point4d
GeometryRenderer::lightPos[] = { Point4d( 0, 0, 1, 1.0 ),
				 Point4d( -0.84, 0, 0.45 ),
				 Point4d( 0.84, 0, 0.45 ),
				 Point4d( 0, -0.84, 0.45 ),
				 Point4d( 0, 0.84, 0.45 ),
				 Point4d( 0, 1, 0 ) };

// Initialize static member variables
const char * GeometryRenderer::version = "1.0";
const char * GeometryRenderer::name = "Geometry Transmission";

const GLfloat 
GeometryRenderer::mat_specular[] = {1.0, 1.0, 1.0, 0.0 };

const GLfloat 
GeometryRenderer::mat_ambient[] = {0.2, 0.2, 0.2, 0.0 };

const GLfloat 
GeometryRenderer::mat_diffuse[] = {0.8, 0.8, 0.8, 0.0 };


unsigned
GeometryRenderer::geomID = 0;

/**
 * Constructor.
 */
GeometryRenderer::GeometryRenderer() : viewParamsChanged(false),
				       calculateNormals( true ),
				       shading( Renderer::SHADING_GOURAUD ),
				       polygonCount( 0 ),
				       fog( false ),
				       lighting( true ),
				       currentGeomColor(0) {
  
  Log::log( ENTER, "[GeometryRenderer::GeometryRenderer] entered" );
  
  geomUI = scinew GeomUI( this );

  // From SCIRun defaults
  View home( Point3d(2.1, 1.6, 11.5), Point3d(.0, .0, .0),
	     Vector3d( 0, 1, 0), 20 );

  geomUI->setHomeView( home );
  geomUI->goHome();

  for (int i = 0; i < NUMBER_OF_LIGHTS; i++ )
    lights[i] = false;
  lights[0] = true; // Turn on at least one light

  Log::log( LEAVE, "[GeometryRenderer::GeometryRenderer] leaving" );

  view_set = false;


}

/**
   * Destructor.
   */
GeometryRenderer::~GeometryRenderer() {

}

/** Computes new near and far planes for the eye and at points, using
      the current bbox. */
void  
GeometryRenderer::compute_depth( const View& v, double &near, double &far ) {  
  Log::log( ENTER, "[GeometryRenderer::compute_depth] entered" );
  near=MAXDOUBLE;
  far=-MAXDOUBLE;
  
  if(bounding_box.valid()) {
    Log::log( DEBUG, "[GeometryRenderer::compute_depth] bounding box valid" );
    // We have something to draw...
    Point3d min(bounding_box.min());
    Point3d max(bounding_box.max());
    Point3d eyep(v.eyep());
    Vector3d dir(v.lookat()-eyep);
    if(dir.length2() < 1.e-6)
      return;
    dir.normalize();
    double d=-Dot(eyep, dir);
    for(int ix=0;ix<2;ix++){
      for(int iy=0;iy<2;iy++){
	for(int iz=0;iz<2;iz++){
	  Point3d p(ix?max.x:min.x,
		    iy?max.y:min.y,
		    iz?max.z:min.z);
	  double dist=Dot(p, dir)+d;
	  near=Min(near, dist);
	  far=Max(far, dist);
	}
      }
    }
    if(near <= 0){
      if(far <= 0){
	// Everything is behind us - it doesn't matter what we do
	near=1.0;
	far=2.0;
      } else {
	near=far*.001;
      }
    }
     Log::log( LEAVE, "[End of GeometryRenderer::compute_depth] leaving" );
    return;
  }
  else {
    Log::log( WARNING, "[GeometryRenderer::compute_depth] bounding box invalid" );
  }
  Log::log( LEAVE, "[End of GeometryRenderer::compute_depth] leaving" );
}

/** Sets up conditions needed (viewing, etc) for geometry rendering */
void  
GeometryRenderer::preRender() {
  Log::log( ENTER, "[GeometryRenderer::preRender] entered, thread id = " + mkString( (int) pthread_self() ) );

  // don't need to change OpenGLcontext -- this function is covered by render function
  
  // make the context current for this thread
  //UserInterface::unlock();
  //UserInterface::lock();
  //OpenGL::mkContextCurrent();

  // TEST_CODE
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::preRender] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::preRender] GL_DEPTH_TEST *not* enabled" );

  glEnable(GL_DEPTH_TEST);

  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::preRender] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::preRender] GL_DEPTH_TEST *not* enabled" );

  // /TEST_CODE
  
  // DEBUG code
  Point3d testEyep( geomUI->getView().eyep() );
  cerr << "eye: <" << testEyep.x << ", " << testEyep.y << ", " << testEyep.z << ">" << endl;

  // Set the view port
  int xres = window_width;
  int yres = window_height;

  Log::log( DEBUG, "[GeometryRenderer::preRender] window_width = " + mkString(window_width) + ", window_height = "  + mkString(window_height) );
  
  glViewport(0, 0, xres, yres);
  geomUI->setRes( window_width, window_height );  
  // Clear the screen?
  
  // Set last view
  geomUI->setLastView();
  
  // Get aspect and fovy
  double aspect=double(xres)/double(yres);
  double fovy=RtoD(2*atan(1.0/aspect*tan(DtoR(geomUI->getView().fov()/2.))));
  
  // Get near and far
  double znear = 0.1, zfar = 1000;

  cerr << "1: znear = " << znear << " zfar = " << zfar << endl;
  compute_depth( geomUI->getView(), znear, zfar);
  cerr << "2: znear = " << znear << " zfar = " << zfar << endl;

  // Set up graphics state
  switch ( shading ) {
  case SHADING_WIREFRAME:
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    break;
  case SHADING_FLAT:
    glShadeModel( GL_FLAT );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    break;
  case SHADING_GOURAUD:
    glShadeModel( GL_SMOOTH );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    break;
  }
  // Check GL errors
  int error;
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[GeometryRenderer::preRender] OpenGL error occurred: " + mkString(gluErrorString(error)) );
  }

  cerr << "3: znear = " << znear << " zfar = " << zfar << endl;

  // Set up perspective
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  cerr << "4: znear = " << znear << " zfar = " << zfar << endl;

  double fov = geomUI->getView().fov();
  //double fov = 90;
  //znear = .1;
  //zfar = 1000;

  // TEST_CODE
  znear = 0.1;
  zfar = 1000;
  // /TEST_CODE

  // Eric's code
  gluPerspective( fov, aspect, znear, zfar);

  Log::log( DEBUG, "[GeometryRenderer::preRender] FOV = " + mkString(fov) );
  Log::log( DEBUG, "[GeometryRenderer::preRender] aspect = " + mkString(aspect) );
  Log::log( DEBUG, "[GeometryRenderer::preRender] znear = " + mkString(znear) );
  Log::log( DEBUG, "[GeometryRenderer::preRender] zfar = " + mkString(zfar) );

  glMatrixMode(GL_MODELVIEW);

  glLoadIdentity();
  
  // Set up modelview
  Point3d eyep( geomUI->getView().eyep() );
  Point3d lookat( geomUI->getView().lookat() );
  Vector3d up( geomUI->getView().up() );
  
  // TEST_CODE
  //eyep = Point3d(0.0, 2.0, -10);
  //lookat = Point3d(0.0, 0.0, 0.0);
  //up = Vector(0, 1, 0);
  // /TEST_CODE
  
  Log::log( DEBUG, "[GeometryRenderer::preRender]eye: <" + mkString(eyep.x) + ", " + mkString(eyep.y) + ", " + mkString(eyep.z) + ">" );
  Log::log( DEBUG, "[GeometryRenderer::preRender]lookat: <" + mkString(lookat.x) + ", " + mkString(lookat.y) + ", " + mkString(lookat.z) + ">" );
  Log::log( DEBUG, "[GeometryRenderer::preRender]up: <" + mkString(up.x) + ", " + mkString(up.y) + ", " + mkString(up.z) + ">" );

  
  gluLookAt(eyep.x, eyep.y, eyep.z,
	    lookat.x, lookat.y, lookat.z,
	    up.x, up.y, up.z);
  
  
  // TEST_CODE
  /*
  gluLookAt(0.0, 0.0, 10.0,
	    0.0, 0.0, 0.0,
	    0.0, 1.0, 0.0);
  */
  // TEST_CODE
  
  // Set up lighting?

  if ( lighting ) {
    glEnable( GL_LIGHTING );
    for ( int i = 0; i < NUMBER_OF_LIGHTS; i++ )
      if ( lights[i] )
	glEnable( (GLenum)( GL_LIGHT0 + i ) );
      else
	glDisable( (GLenum)( GL_LIGHT0 + i ) );
  }
  else {
    glDisable( GL_LIGHTING );
  }

  // Set up fog?
  if ( fog ) {
    float color[4] = {0,0,0,1};
    glEnable( GL_FOG );
    glFogi(GL_FOG_MODE,GL_LINEAR);
    glFogf(GL_FOG_START,float(znear));
    glFogf(GL_FOG_END,float(zfar));
    glFogfv( GL_FOG_COLOR, color );
  }
  else
    glDisable( GL_FOG );

  //OpenGL::finish();
  //UserInterface::unlock();

  // TEST_CODE

  double modelview[16];
  double projMatrix[16];
  int viewPort[4];
  
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
  glGetIntegerv(GL_VIEWPORT, viewPort);

  Log::log( DEBUG, "[GeometryRenderer::preRender] modelview:" );
  for ( int i = 0; i < 16; i++ )
  {
    Log::log( DEBUG, "[GeometryRenderer::preRender]" + mkString(modelview[ i ]) );
  }

  Log::log( DEBUG, "[GeometryRenderer::preRender] projMatrix:" );
  for ( int i = 0; i < 16; i++ )
  {
    Log::log( DEBUG, "[GeometryRenderer::preRender] " + mkString(projMatrix[ i ]) );
  }

  Log::log( DEBUG, "[GeometryRenderer::preRender] viewPort: (" + mkString(viewPort[0]) + ", " + mkString(viewPort[1]) + ", " + mkString(viewPort[2]) + ") " );

  // Check GL errors
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[ZTexRenderer::render] OpenGL error occurred, error code " + error);
  }
  // /TEST_CODE
  
  Log::log( LEAVE, "[GeometryRenderer::preRender] leaving" );
}

/** Renders the image into the screen. */
void 
GeometryRenderer::render( const bool doFPS ) {
  Log::log( ENTER, "[GeometryRenderer::render] entered, thread id = " + mkString( (int) pthread_self() ) );

  // the context is already updated for this function before it is called
  
  // make the context current for this thread
  //Log::log( DEBUG, "[GeometryRenderer::render] locking user interface" );
  //UserInterface::unlock();
  //UserInterface::lock();
  
  //if( OpenGL::mkContextCurrent() < 0)
  //{
  //  Log::log( ERROR, "[GeometryRenderer::render] couldn't make context current" );
  // return;
  //}
  
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE)
    Log::log( DEBUG, "[GeometryRenderer::render] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::render] GL_DEPTH_TEST *not* enabled" );

      
  // TEST_CODE
  //glEnable( GL_LIGHTING );
  // make light a bit darker than default
  //GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 }; 
  // glLightfv( GL_LIGHT0, GL_DIFFUSE, light_diffuse );
  //glEnable( GL_DEPTH_TEST );
  //glEnable(GL_NORMALIZE);
  //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  // /TEST_CODE

  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE)
    Log::log( DEBUG, "[GeometryRenderer::render] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::render] GL_DEPTH_TEST *not* enabled" );
  
  WallClockTimer timer;

  timer.start();
  
  // Set up rendering stuff
  preRender();
  
  // Draw our display lists
  Log::log( DEBUG, "[GeometryRenderer::render] Drawing " + mkString(displayLists.size()) +
	    " display lists" );
  for ( unsigned i = 0; i < displayLists.size(); i++ )
    if ( displayLists[i].enabled )
      glCallList( displayLists[i].list );
  
  // Draw annotations
  Log::log( DEBUG, "[GeometryRenderer::render] Drawing " + mkString(collaborationItemList.size()) +
	    " annotations" );
  if ( collaborationItemList.size() > 0 ) {
    if ( lighting )
      glDisable( GL_LIGHTING );
    for ( unsigned i = 0; i < collaborationItemList.size(); i++ ) {
      collaborationItemList[i].draw();
      UserInterface::getHelper()->checkErrors();
    }
    if ( lighting )
      glEnable( GL_LIGHTING );
  }

  Log::log( DEBUG, "[GeometryRenderer::render] after annotations" );
  
  timer.stop();
  fps = 1. / timer.time();
  
  UserInterface::getHelper()->setFPS( fps );

  //OpenGL::finish();
  //UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::render] leaving" );
}

/**
 * Receives a mouse event from the rendering window.
 *
 * @param       me           Mouse event.
 */
void 
GeometryRenderer::mouseMove( MouseEvent me ) {
  Log::log( ENTER, "[GeometryRenderer::mouseMove] entered" );

  // this function doesn't have any OpenGL calls, so the context doesn't need to be made current
  
  // make the context current for this thread
  //UserInterface::lock();
  //OpenGL::mkContextCurrent();
  
  if ( me.button == 'L' )
  {
    geomUI->translate( me.action, me.x, me.y );
  }
  else if ( me.button == 'M' )
  {
    geomUI->rotate( me.action, me.x, me.y, 0 ); // Add me.time
  }
  else if ( me.button == 'R' )
  {
    geomUI->scale( me.action, me.x, me.y );
  }
  
  draw();

  /*
  // send mouse event data to server *JS*
  MouseMove *m = scinew MouseMove();
  m->setMove( me.x, me.y, me.button, me.action );
  m->finish();

  NetInterface::getInstance().sendDataToServer( m );
  */

  //OpenGL::finish();
  //UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::mouseMove] leaving" );
}

   
/** Initializes OpenGL for geometry rendering */
void  
GeometryRenderer::initialize() {
  Log::log( ENTER, "[GeometryRenderer::initialize] entered, thread id = " + mkString( (int) pthread_self() ) );
  /** Set up lights */

  // make the context current for this thread
  //UserInterface::unlock();
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  cerr << "Setting up lights" << endl;
  glShadeModel( GL_SMOOTH ); // Gouraud shading

  cerr << "Setting material properties" << endl;
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

  cerr << "Before light colors loop" << endl;
  for ( int i = 0; i < numLightColors; i++ ) {
    glLightfv( (GLenum)(GL_LIGHT0+i), GL_POSITION, lightPos[i].array());
    glLightfv( (GLenum)(GL_LIGHT0+i), GL_DIFFUSE, lightColors[i].array());
    //glLightfv( (GLenum)(GL_LIGHT0+i), GL_AMBIENT, lightColors[i].array());
  }

  cerr << "after light colors loop" << endl;
  
  if ( lighting )
  {
    glEnable(GL_LIGHTING);
  }
  
  cerr << "Turn on light 0" << endl;

  // TEST_CODE
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST *not* enabled" );

  if( glIsEnabled( GL_LIGHTING ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING *not* enabled" );

  if( glIsEnabled( GL_NORMALIZE ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE *not* enabled" );
  // /TEST_CODE
  
  // Initially turn on light 0
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  //GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 }; 
  //glLightfv( GL_LIGHT0, GL_DIFFUSE, light_diffuse );
  
  cerr << "Set up model view" << endl;
  /* Set up initial look at stuff on the model view matrix stack */
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Check GL errors
  int error;
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[GeometryRenderer::initialize] OpenGL error occurred: " + mkString(gluErrorString(error)) );
  }

  // TEST_CODE
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST *not* enabled" );

  if( glIsEnabled( GL_LIGHTING ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING *not* enabled" );

  if( glIsEnabled( GL_NORMALIZE ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE *not* enabled" );
  // /TEST_CODE
  
    /*
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

   if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_DEPTH_TEST *not* enabled" );

  if( glIsEnabled( GL_LIGHTING ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_LIGHTING *not* enabled" );

  if( glIsEnabled( GL_NORMALIZE ) == GL_TRUE )
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::initialize] GL_NORMALIZE *not* enabled" );
  */
  
  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::initialize] leaving" );
}

void
GeometryRenderer::addGeometry( float *vertices, const int numVertices,
			       const bool replace ) {

  Log::log( ENTER, "[GeometryRenderer::addGeometry] entered" );

  // make the context current for this thread
  //UserInterface::unlock();
  UserInterface::lock();
  OpenGL::mkContextCurrent();

  // TEST_CODE
  
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE)
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] GL_DEPTH_TEST *not* enabled" );

  glEnable( GL_DEPTH_TEST );
  
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE)
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] GL_DEPTH_TEST *not* enabled" );

  // /TEST_CODE
  
  if ( numVertices % 3 != 0 ) {
    Log::log( ERROR, "[GeometryRenderer::addGeometry] Vertex count not a multiple of 3. Got " +
	      mkString( numVertices ) );
    return;
  }
  Log::log( MESSAGE, "[GeometryRenderer::addGeometry] Adding " + mkString( numVertices / 3 ) + " triangles" );
  
  int numNumbers = numVertices * 3;
  
  BBox b;
  GLuint display_list = glGenLists(1);
  if ( display_list != 0 ) {
    geomObj g( display_list, geomColors[ currentGeomColor % numGeomColors ],
	       geomID++ );
    currentGeomColor++;
    
    // New List
    glNewList( display_list, GL_COMPILE );

    // Set the color
    glColor3fv( g.color.array() );

    
    // Set the material properties
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
		  g.color.array() );
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] Did gl stuff" );
    // Do the triangles - calculate normals too, if needed
    if ( calculateNormals ) {
      Vector3d normal;
      
      glBegin( GL_TRIANGLES );
      Log::log( DEBUG, "[GeometryRenderer::addGeometry] NumNumbers: " + mkString( numNumbers ) );
      for ( int i = 0; i < numNumbers; i += 9 ) {
	/*cerr << "Triangle: (" << vertices[ i ] << ", " << vertices[ i+1 ] <<
	  ", " << vertices[ i+2 ] << "), (" << vertices[ i+3 ] << ", " <<
	  vertices[ i+4 ] <<  ", " << vertices[ i+5 ] << "), (" <<
	  vertices[ i+6 ] << ", " << vertices[ i+7 ] <<  ", " <<
	  vertices[ i+8 ] << ")" << endl;*/
	normal = mkNormal( &vertices[ i ], 0, true );
	/*cerr << "Normal: (" << normal.x << ", " << normal.y << ", " <<
	  normal.z << ") " << endl;*/
	glNormal3f( normal.x, normal.y, normal.z );
	glVertex3fv( vertices + i );
	glVertex3fv( vertices + i+3 );
	glVertex3fv( vertices + i+6 );

	// Build the bounding box
	b.extend( Point3d( vertices[i], vertices[i+1], vertices[i+2] ) );
	b.extend( Point3d( vertices[i+3], vertices[i+4], vertices[i+5] ) );
	b.extend( Point3d( vertices[i+6], vertices[i+7], vertices[i+8] ) );  
      }
      glEnd();
    }
    else {
      glBegin( GL_TRIANGLES );
      for ( int i = 0; i < numNumbers; i+=9 ) {
	glVertex3fv( vertices + i );
	glVertex3fv( vertices + i+3 );
	glVertex3fv( vertices + i+6 );
      }
      glEnd();
    }

    // End the list
    glEndList();
    
    displayLists.push_back( g );

    Log::log( DEBUG, "[GeometryRenderer::addGeometry] New geom bounding box: " + b.toString() );
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] Global bounding box was " + bounding_box.toString() );
    bounding_box.extend(b);
    Log::log( DEBUG, "[GeometryRenderer::addGeometry] Global bounding box is now " + bounding_box.toString() );
  }
  else {
    Log::log( ERROR, "[GeometryRenderer::addGeometry] Unable to allocate display list for geom renderer!" );
  }
  
  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::addGeometry] leaving" );
}


/** Sets testing geometry */
void 
GeometryRenderer::setTestGeometry() {

  Log::log( ENTER, "[GeometryRenderer::setTestGeometry] entered" );

  // make the context current for this thread
  //UserInterface::lock();
  //OpenGL::mkContextCurrent();
  
#if 1
  float triangle[] = { -5, -2.5, 0,
		       0, 2.5, 0,
		       5, -2.5, 0 };

  // These three triangles should look like a three-sided pyramid pointing up along the y axis
  
  // base traingle
  float t1[] = { -3, -3, 0,
		 3, -3, 0,
		 0, -3, -5 };

  // side triangles
  float t2[] = { 0, 2, -2,
		 0, -3, -5,
		 3, -3, 0 };
  float t3[] = { 0, 2, -2,
		 -3, -3, 0,
		 0, -3, -5 };
  float t4[] = { 0, 2, -2,
		 3, -3, 0,
		 -3, -3, 0 };

  /*
  
  float t1[] = { -5, -5, -2.5,
		 0, -5, -5,
		 5, -5, -2.5 };
  float t2[] = { 0, 5, 0,
		 0, -5, -5,
		 -5, -5, -2.5 };
  float t3[] = { 0, 5, 0,
		 5, -5, -2.5,
		 0, -5, -5 };
  float t4[] = { -5, -5, -2.5,
		 5, -5, -2.5,
		 0, 5, 0 };
  */
  
  //addGeometry( triangle, 9, true );
  addGeometry( t1, 3, true );
  addGeometry( t2, 3, true );
  addGeometry( t3, 3, true );
  addGeometry( t4, 3, true );
#else
  GLuint list = glGenLists(1);
  if ( list != 0 ) {
    geomObj g( list, Color( 255, 255, 255 ), geomID++ );
    
    // New List
    glNewList( list, GL_COMPILE_AND_EXECUTE );

    // Set the color
    glColor3fv( g.color.array() );

    // Set the material properties
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
		  g.color.array() );
    
    // Do the triangles - calculate normals too, if needed
    glutSolidSphere( 2, 40, 40 );

    // End the list
    glEndList();
    
    displayLists.push_back( g );
  }
  else {
    Log::log( ERROR, "[GeometryRenderer::setTestGeometry] Unable to allocate display list for geom renderer!" );
  }
#endif

  //OpenGL::finish();
  //UserInterface::unlock(); 

  Log::log( LEAVE, "[GeometryRenderer::setTestGeometry] leaving" );
}

/**
   * Enables/Disables the geometric object (including annotations)
   * with the given ID. 
   * 
   * @param     ID          ID of the object.
   * @param     enable      True to enable the object. False to
   *                        disable it.
   */
void 
GeometryRenderer::enableGeometryAt( const string ID, const bool enable ) {
  Log::log( ENTER, "[GeometryRenderer::enableGeometryAt] entered" );
  Log::log( LEAVE, "[GeometryRenderer::enableGeometryAt] leaving" );
}

/**
 * Deletes the geometric object (including annotations) with the
 * given ID.
 *
 * @param ID  ID of the object.
 *
 */
void 
GeometryRenderer::deleteAnnotation( const string ID ) { 
  Log::log( ENTER, "[GeometryRenderer::deleteAnnotation] entered" );

  // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  if ( startsWith( ID, "Geom" ) ) {
    int geomID = atoi( ID.substr(4) );
    Log::log( DEBUG, "[GeometryRenderer::deleteAnnotation] Deleting geometry " + ID +
	      " ID = " + mkString( geomID ) );
    vector<geomObj>::iterator i;
    
    for ( i = displayLists.begin(); i != displayLists.end(); i++ )
      if ( i->ID == geomID ) {
	glDeleteLists( i->list, 1 );
	displayLists.erase(i);
	return;
      }
  }
  else 
    Renderer::deleteAnnotation( ID );

  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::deleteAnnotation] leaving" );
}

/**
 * Returns the names (IDs) of all the geometric objects currently in
 * the scene.
 *
 * @return    A list of the IDs of all the geometric objects in the
 *            scene. 
 */
vector<string> 
GeometryRenderer::getGeometryNames() { 
  Log::log( ENTER, "[GeometryRenderer::getGeometryNames] entered" );

  vector<string> names( displayLists.size() );

  for ( unsigned j = 0; j < displayLists.size(); j++ )
    names[j] = string("Geom ") + mkString( displayLists[j].ID );

  Log::log( LEAVE, "[GeometryRenderer::getGeometryNames] leaving" );
  return names;
}

/**
 * Goes to the 'home view' - a view that enables us to look at the center
 * of the object, far enough away that we get all of the geometry in our
 * viewing frustum.
 */
void 
GeometryRenderer::homeView() { 
  Log::log( ENTER, "[GeometryRenderer::homeView] entered" );

  // make the context current for this thread
  //UserInterface::lock();
  //OpenGL::mkContextCurrent();
  
  geomUI->goHome();
  draw();
  Log::log( LEAVE, "[GeometryRenderer::homeView] leaving" );
}
 
/**
 * Called when we receive another viewing frame from the server.
 * 
 * @param       message           ViewFrame message.
 */
void 
GeometryRenderer::getViewFrame( MessageData * message ) {  
  Log::log( ENTER, "[GeometryRenderer::getViewFrame] entered" );
  ViewFrame *vf = (ViewFrame *)(message->message);
  char * unconvertedData=NULL;
  bool replace = false;

  // doesn't make any OpenGL calls directly, so doesn't need to make context current
  
  Log::log( DEBUG, "[GeometryRenderer::getViewFrame] Got a gr view frame!" );

  /* Get the original size of the data */
  int origSize = vf->getFrameSize();
  if ( origSize < 0 ) 
    return;

  /*  GET RENDERING DATA */
  if ( vf->isReplaceSet() ) replace = vf->isReplace();
  int numVertices = vf->getVertexCount();
  int numIndices  = vf->getIndexCount();
  int numPolygons = vf->getPolygonCount();

  Log::log( DEBUG, "[GeometryRenderer::getViewFrame] Got " + mkString(numVertices) + " vertices, " +
	    mkString(numIndices) + " indices, and " + mkString(numPolygons) +
	    " polygons");
  
  /* Decompress the data if needs be */
  if ( compressionMode != NONE ) {
    Log::log( DEBUG, string("[GeometryRenderer::getViewFrame] Decompressing viewframe for ") +
	      mkString(message->size) +
	      " bytes. Decompressed length is " + mkString(origSize) );
    unconvertedData = decompressFrame( message->data, 0,
				       message->size, origSize );
    delete message->data; // Because we do not need it anymore
    if ( unconvertedData == NULL ) {
      Log::log( ERROR, "[GeometryRenderer::getViewFrame] Error decompressing image data!" );
      return;
    }
  }
  else {
    // Sanity test. Barf if size != origsize
    if ( message->size != origSize ) {
      Log::log( ERROR, string("[GeometryRenderer::getViewFrame] Uncompressed frame size error! We need ") +
		mkString( origSize ) + ", and got " +
		mkString( message->size ) );
      return;
    }
    Log::log( DEBUG, "[GeometryRenderer::getViewFrame] Using uncompressed data." );
    unconvertedData = message->data;
  }

  /* Now, our data falls into several cases: */
  
  /* Triangles only, not indexed. */
  if ( 1 || numPolygons >= 0 && !vf->isIndexed() ) {

    /* Convert data */
    int floatLen = origSize / sizeof(float); // Number of floats
    float * floatData = scinew float[ floatLen ];

    // TMP
    if (0) {
      float * fd = (float *)unconvertedData;
      FILE * f = fopen( "trisb.out", "w" );
      for ( int i = 0; i < floatLen; i+=3 ) {
	fprintf(f, "Vertex %d: (%0.2f, %0.2f, %0.2f)\n",
		( i / 3 ) + 1, fd[ i ], fd[ i+1 ],
		fd[ i+2 ] );
      }
      fclose(f);
    }
    // /TMP
    
    Log::log( DEBUG, "[GeometryRenderer::getViewFrame] Converting " + mkString( floatLen ) +
	      " floats, for a total size of " +
	      mkString(floatLen * sizeof(float)) );
    Log::log( DEBUG, "[GeometryRenderer::getViewFrame] First vertex before: " +
	      mkString( ((float*)unconvertedData)[0] ) );
    
    float * undata = (float *)unconvertedData;
    DataDescriptor d = SIMPLE_DATA( FLOAT_TYPE, 1 );
    for ( int q = 0; q < floatLen; q++ ) {
      ConvertHostToNetwork( (void*)&floatData[q],
			    (void*)&undata[q],
			    &d,
			    1 );
    }
    Log::log( DEBUG, "[GeometryRenderer::getViewFrame] First vertex after: " + mkString( floatData[0] ) );
    
    // Sanity check: #vertices should = 1/3 length of float data
    if ( numVertices != floatLen / 3 ) {
      Log::log( ERROR, 
		"[GeometryRenderer::getViewFrame] Number of vertices present does not equal number claimed: " +
		mkString( floatLen / 3 ) + " / " + mkString(numVertices) );
      
      // Match it to what we have, anyway...
      numVertices = floatLen / 3; // TEST_CODE
    }

    // TMP
    if ( 0 ) {
      FILE * f = fopen( "tris.out", "w" );
      for ( int i = 0; i < floatLen; i+=3 ) {
	fprintf(f, "Vertex %d: (%0.2f, %0.2f, %0.2f)\n",
		( i / 3 ) + 1, floatData[ i ], floatData[ i+1 ],
		floatData[ i+2 ] );
      }
      fclose(f);
    }
    // /TMP

    // Add the geometry 
    addGeometry( floatData, numVertices, replace );

    // TEST_CODE
    //setTestGeometry();
    // /TEST_CODE
  }
  else {
    Log::log( ERROR, "[GeometryRenderer::getViewFrame] Unsupported geometry format!" );
    return;
  }

  // Redraw the screen!
  Log::log( DEBUG, "[GeometryRenderer::getViewFrame] Redrawing after geom view frame" );

  if ( (void *)redraw != NULL )
    redraw();
  Log::log( LEAVE, "[GeometryRenderer::getViewFrame] leaving" );
}


/**
 * Called when we receive an acknowledgement of our change of viewing
 * methods (or viewing parameters) from the server.
 * 
 * @param       message           Message from the server.
 */
void 
GeometryRenderer::setViewingMethod( MessageData * message ) {  
  Log::log( ENTER, "[GeometryRenderer::setViewingMethod] entered, thread id = " + mkString( (int) pthread_self() ) );

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
  
  // TEST_CODE
  Point3d eyep( geomUI->getView().eyep() );
  cerr << "eye: <" << eyep.x << ", " << eyep.y << ", " << eyep.z << ">" << endl;

  view_set = true;
  // /TEST_CODE
  
  SetViewingMethod *svm = (SetViewingMethod *)(message->message);  

  // update viewing info for defined parameters in svm message
  Point3d eyePoint, lookAtPoint;
  Vector upVector;
  double FOV;

  // check all params to see if they are defined
  // if they are not defined, use old value

  if((eyePoint = svm->getEyePoint()).set()){
    //Log::log( DEBUG, "[GeometryRenderer::setViewingMethod Setting new eye point to (" + mkString(eyePoint.x) + ", " + mkString(eyePoint.y) + ", " + mkString(eyePoint.z) + ")" );
  } 
  else{
    cerr << "[GeometryRenderer::setViewingMethod]  ERROR: NULL eye point!" << endl;
    Log::log( ERROR, "[GeometryRenderer::setViewingMethod] NULL eye point, using old value" );
    eyePoint = geomUI->getView().eyep();
  
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new eye point to (" + mkString(eyePoint.x) + ", " + mkString(eyePoint.y) + ", " + mkString(eyePoint.z) + ")" );
  }
  // TEST_CODE
  /*
    eyePoint.x = 0.0;
    eyePoint.y = 0.0;
    eyePoint.z = 10;
  */
  // /TEST_CODE

  if((lookAtPoint = svm->getLookAtPoint()).set()){
    //Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new lookat point to (" + mkString(lookAtPoint.x) + ", " + mkString(lookAtPoint.y) + ", " + mkString(lookAtPoint.z) + ")" );
  }
  else{
    cerr << "[GeometryRenderer::setViewingMethod]  ERROR: NULL lookat point!" << endl;
    Log::log( ERROR, "[GeometryRenderer::setViewingMethod] NULL lookat point, using old value" );
    lookAtPoint = geomUI->getView().lookat();
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new lookat point to (" + mkString(lookAtPoint.x) + ", " + mkString(lookAtPoint.y) + ", " + mkString(lookAtPoint.z) + ")" );
  }

  upVector = svm->getUpVector();
  Point3d tempUpVector(upVector.x, upVector.y, upVector.z);

  //cerr << "upVector: <" << upVector.x << ", " << upVector.y << ", " << upVector.z << ">" << endl;

  if(tempUpVector.set()){
    //Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new up vector to (" + mkString(tempUpVector.x) + ", " + mkString(tempUpVector.y) + ", " + mkString(tempUpVector.z) + ")" );
  }
  else{
    cerr << "[GeometryRenderer::setViewingMethod]  ERROR: NULL up vector!" << endl;
    Log::log( ERROR, "[GeometryRenderer::setViewingMethod] NULL up vector, using old value" );
    upVector = geomUI->getView().up();
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new up vector to (" + mkString(upVector.x) + ", " + mkString(upVector.y) + ", " + mkString(upVector.z) + ")" );
  } 

  if((FOV = svm->getFOV()) != DBL_MIN){
    //Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new fov to " + mkString(FOV));
  }
  else{
    cerr << "[GeometryRenderer::setViewingMethod]  ERROR: NULL fov!" << endl;
    Log::log( ERROR, "[GeometryRenderer::setViewingMethod] NULL fov, using old value" );
    FOV = geomUI->getView().fov();
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new fov to " + mkString(FOV));
  } 



  //setEyepoint( svm->getEyePoint() );
  //setLookAtPoint( svm->getLookAtPoint() );
  //setUpVector( svm->getUpVector() );
  //geomUI->getView() = View( svm->getEyePoint(),
  //			    svm->getLookAtPoint(),
  //			    svm->getUpVector(),
  //			    svm->getFOV() );

  geomUI->getView() = View( eyePoint,
  			    lookAtPoint,
  			    upVector,
  			    FOV );



  // TEST_CODE
  Point3d testEyep( geomUI->getView().eyep() );
  cerr << "eye: <" << testEyep.x << ", " << testEyep.y << ", " << testEyep.z << ">" << endl;
  // /TEST_CODE
  
  //setPerspective( svm->getFOV(),
  //svm->getNear(),
  //svm->getFar() );
  
  
  // Switch fov, near, and/or far
  // FOV is already set

  if(svm->getNear() != DBL_MIN ) {
    near = svm->getNear();
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new near plane to " + mkString(near));
  }
  if ( svm->getFar() != DBL_MIN ) {
    far = svm->getFar();
    Log::log( DEBUG, "[GeometryRenderer::setViewingMethod] Setting new far plane to " + mkString(far));
  }  

  // For building ztex objects
  near = svm->getNear();
  far = svm->getFar();
  FOV = svm->getFOV();
  
  if ( (void *)redraw != NULL )
    redraw();

  
  Log::log( LEAVE, "[GeometryRenderer::setViewingMethod] leaving" );
}

/**
   * Determines the presence of fog. (on/off)
   *
   * @param    mode        Whether we want fog.
   */
void 
GeometryRenderer::setFog(const bool show) {

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
    
  Log::log( ENTER, "[GeometryRenderer::setFog] entered" );
  fog = show;
  draw();

  Log::log( LEAVE, "[GeometryRenderer::setFog] End of GeometryRenderer::setFog" );
}

/**
   * Turns lighting on/off.
   *
   * @param      lighting       True to enable lighting; false to disable. 
   */
void 
GeometryRenderer::setLighting(const bool lighting) {
  Log::log( ENTER, "[GeometryRenderer::setLighting] entered" );

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
  
  this->lighting = lighting;
  draw();

  Log::log( LEAVE, "[GeometryRenderer::setLighting] leaving" );
}

/**
   * Turns on/off a particular light.
   *
   * @param whichLight      Which light to turn on (0 -> 
   *                        (NUMBER_OF_LIGHTS - 1))
   * @param show            Turn on (true) or off (false).
   */
void 
GeometryRenderer::setLight(const int whichLight, const bool show) {
  Log::log( ENTER, "[GeometryRenderer::setLight] entered" );

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
  
  if ( whichLight >= NUMBER_OF_LIGHTS ) return;

  lights[ whichLight ] = show;
  draw();
  Log::log( LEAVE, "[GeometryRenderer::setLight] leaving" );  
}

/**
   * Sets the current shading type for rendering.
   *
   * @param        shadingType     Desired shading type.
   *
   */
void 
GeometryRenderer::setShadingType(const int shadingType) {  
  Log::log( ENTER, "[GeometryRenderer::setShadingType] entered" );

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
  
  switch ( shadingType ) {
   case SHADING_FLAT:
    Log::log( MESSAGE, "[GeometryRenderer::setShadingType] Changing to flat shading" );
    break;
  case SHADING_GOURAUD:
    Log::log( MESSAGE, "[GeometryRenderer::setShadingType] Changing to gouraud shading" );
    break;
  case SHADING_WIREFRAME:
    Log::log( MESSAGE, "[GeometryRenderer::setShadingType] Changing to wireframe shading" );
    break;
  default:
    Log::log( ERROR, "[GeometryRenderer::setShadingType] Unknown shading type in geomRenderer: " +
	      mkString( shadingType ) );
    return;
  }
  shading = shadingType;
  draw();

  Log::log( LEAVE, "[GeometryRenderer::setShadingType] leaving" ); 
}

/**
   * Adds a pointer at the given location, with the given rotations around
   * the two relevant axes.
   *
   * @param    point        Location of pointer.
   * @param    zRotation    Rotation in radians around the Z axis.
   *
   * @return                An ID for this pointer.
   */
string 
GeometryRenderer::addPointer( const Point3d point, const double zRotation ) {
  Log::log( ENTER, "[GeometryRenderer::addPointer] entered" );

  // don't need to lock and make context current because this function doesn't have any OpenGL calls
  
  // Get the point in world coords
  Point3d newPoint = projectPoint( point );
  
  // Add the pointer using the new params
  string ID = 
    Renderer::addPointer( newPoint, zRotation );
  getID( ID )->setPointerSize( 2.0 ); // Make the pointer a bit smaller!

  Log::log( LEAVE, "[GeometryRenderer::addPointer] leaving" ); 
  return ID;
}

/**
 * Adds the text at the given coordinates.
 *
 * @param     point      Location of the text
 * @param     text       Text to add.
 *
 * @return               An ID for this text.
 */
string  
GeometryRenderer::addText( const Point3d point, const string &text ) {
  Log::log( ENTER, "[GeometryRenderer::addText] entered" );

  // make the context current for this thread
  // Get the point in world coords
  Point3d newPoint = projectPoint( point );
  
  // Add the text using the new params
  string ID = 
    Renderer::addText( newPoint, text );

  Log::log( LEAVE, "[GeometryRenderer::addText] leaving" ); 
  return ID;
}

/**
 * Adds another segment to the drawing with the given ID.
 *
 * @param        point        Location of the next line segment.
 * @param        ID           Which drawing to add the segment to.
 */
void    
GeometryRenderer::addDrawingSegment( const Point3d point,
				     const string &ID ) {
  Log::log( ENTER, "[GeometryRenderer::addDrawingSegment] entered" ); 
  // Get the point in world coords
  Point3d newPoint = projectPoint( point );
  
  // Add the point using the new params
  Renderer::addDrawingSegment( newPoint, ID );
  Log::log( LEAVE, "[GeometryRenderer::addDrawingSegment] leaving" ); 
}


/** Projects the point from screen to world space. */
Point3d   
GeometryRenderer::projectPoint( const Point3d point ) {
  Log::log( ENTER, "[GeometryRenderer::projectPoint] entered" ); 
  /* We have to be clever. We are given x and y in screen coordinates.
     However, we need them in OpenGL space; thus, we must unproject these
     coords to z = 0, and add the pointer with this orientation.
  */
  
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  // Grab OpenGL params
  GLfloat model[16], proj[16];
  GLint view[4];
  
  glGetFloatv( GL_MODELVIEW_MATRIX, model );
  glGetFloatv( GL_PROJECTION_MATRIX, proj );
  glGetIntegerv( GL_VIEWPORT, view );

  GLdouble world[3];

  GLdouble m[16], p[16];


  // Copy floats to doubles. 
  for ( int i = 0; i < 16; i++ ) {
    m[i] = model[i]; p[i] = proj[i];
  }
  
  // Project the center of the bounding box, and use that Z coordinate.
  Log::log( DEBUG, "[GeometryRenderer::projectPoint] Bounding box is " + bounding_box.toString() );
  Point3d midpoint = bounding_box.middle();
  gluProject( midpoint.x, midpoint.y, midpoint.z,
	      m, p, view, &world[0], &world[1], &world[2] );
  Log::log( DEBUG, "[GeometryRenderer::projectPoint] Midpoint of bounding box " + midpoint.toString() +
	    " maps to window Z of " + mkString(world[2]) );
  
  Point3d adjusted_point( point.x,
			  window_height,
			  point.y,
			  world[2] );
  if ( gluUnProject( adjusted_point.x, adjusted_point.y,
		     adjusted_point.z,
		     m,
		     p,
		     view,
		     &world[0],
		     &world[1],
		     &world[2] ) == GL_TRUE ) {
    // Flip the Y coordinate.
    world[1] = -world[1];

    Point3d newPoint( world[0], world[1], world[2] );
    
    // Unproject went okay
    Log::log( DEBUG, "[GeometryRenderer::projectPoint] Window coords from " +
	      adjusted_point.toString() + " to " +
	      newPoint.toString() );
    
    //UserInterface::unlock();
    return newPoint;
  }
  else {
    // Uninvertible matrix. Doh! :(
    Log::log( ERROR, "[GeometryRenderer::projectPoint] Unable to unproject window coord vertex " +
	      point.toString() );
    //UserInterface::unlock();
    return Point3d();
  }

  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[GeometryRenderer::projectPoint] leaving" ); 
}


/** 
 * Calculates normal for the vertices in 'vertices' in a CCW direction.
 * If the flip parameter is true, it flips the normal.
 */
Vector3d 
GeometryRenderer::mkNormal( float* vertices, const int offset,
			    const bool flip ) {  
  //Log::log( ENTER, "[GeometryRenderer::mkNormal] entered" );

  /*
  Vector3d v1( vertices[ offset + 3 ] - vertices[ offset + 0 ], 
	       vertices[ offset + 4 ] - vertices[ offset + 1 ], 
	       vertices[ offset + 5 ] - vertices[ offset + 2 ] );
  Vector3d v2( vertices[ offset + 3 ] - vertices[ offset + 6 ], 
	       vertices[ offset + 4 ] - vertices[ offset + 7 ], 
	       vertices[ offset + 5 ] - vertices[ offset + 8 ] );
  */
  
  Vector3d v1( vertices[ offset + 0 ] - vertices[ offset + 3 ], 
	       vertices[ offset + 1 ] - vertices[ offset + 4 ], 
	       vertices[ offset + 2 ] - vertices[ offset + 5 ] );
  Vector3d v2( vertices[ offset + 6 ] - vertices[ offset + 3 ], 
	       vertices[ offset + 7 ] - vertices[ offset + 4 ], 
	       vertices[ offset + 8 ] - vertices[ offset + 5 ] );
  
  Vector3d normal = Vector3d::cross( v1, v2 );
  normal.normalize();

  if ( flip )
  {
    normal.negate();
    //Log::log( DEBUG, "[GeometryRenderer::mkNormal] flipping normal" );
  }

  //Log::log( DEBUG, "[GeometryRenderer::mkNormal] Calculated normal: " +
  //	      normal.toString() );
   
  //Log::log( LEAVE, "[GeometryRenderer::mkNormal] leaving" );
  return normal;
}

/** Enables/Disables the geometry with the given ID. */
void 
GeometryRenderer::enableGeom( const string ID, const bool enable ) {  
  Log::log( ENTER, "[GeometryRenderer::enableGeom] entered" );
  Log::log( LEAVE, "[GeometryRenderer::enableGeom] leaving" );
 
}

/** Enables/Disables the pointer with the given ID. */
void 
GeometryRenderer::enablePointer( const string ID, const bool enable ) {  
  Log::log( ENTER, "[GeometryRenderer::enablePointer] entered" );
  Log::log( LEAVE, "[GeometryRenderer::enablePointer] leaving" );
}

/** Enables/Disables the text with the given ID. */
void 
GeometryRenderer::enableText( const string ID, const bool enable ) {  
  Log::log( ENTER, "[GeometryRenderer::enableText] entered" );
  Log::log( LEAVE, "[GeometryRenderer::enableText] leaving" );
}
  
/** Enables/Disables the drawing with the given ID. */
void 
GeometryRenderer::enableDrawing( const string ID, const bool enable ) { 
  Log::log( ENTER, "[GeometryRenderer::enableDrawing] entered" );
  Log::log( LEAVE, "[End of GeometryRenderer::enableDrawing] leaving" );
}

/**
 * Enables or disables the annotation with the given ID.
 *
 * @param    ID         ID of the annotation
 * @param    enable     True to enable the annotation; false to disable 
 */
void 
GeometryRenderer::enableAnnotation( const string ID, const bool enable ) {
  Log::log( ENTER, "[GeometryRenderer::enableAnnotation] entered" );
  if ( startsWith( ID, "Geom" ) ) {
    int geomID = atoi( ID.substr(4) );
    Log::log( DEBUG, (enable ? "[GeometryRenderer::enableAnnotation] Enabling" : "Disabling") +
	      string(" geometry ") + ID + " ID = " + mkString( geomID ) );
    for ( unsigned i = 0; i < displayLists.size(); i++ )
      if ( displayLists[i].ID == geomID ) {
	displayLists[i].enabled = enable;
	return;
      }
  }
  else 
    Renderer::enableAnnotation( ID, enable );
  Log::log( LEAVE, "[GeometryRenderer::enableAnnotation] leaving" );
}

vector<string>
GeometryRenderer::getInfo() {
  Log::log( ENTER, "[GeometryRenderer::getInfo] entered" );

  vector<string> a = getAnnotationIDs();
  vector<string> g = getGeometryNames();
  
  // Reserve needed space.
  vector<string> info( a.size() + g.size() );
  
  // Annotations, then geom
  unsigned j;
  for ( j = 0; j < a.size(); j++ )
    info[j] = a[j];

  for ( unsigned i = 0; i < g.size(); i++, j++ )
    info[j] = g[i];

  Log::log( LEAVE, "[GeometryRenderer::getInfo] leaving" ); 
  return info;
}
  


}
