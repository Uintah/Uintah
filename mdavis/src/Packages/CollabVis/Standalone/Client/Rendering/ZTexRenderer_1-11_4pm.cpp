/*
 *
 * ZTexRenderer: Renderer dedicated to ZTex rendering.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */
#include <Rendering/ZTexRenderer.h>
#include <Network/NetConversion.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <UI/UserInterface.h>
#include <UI/uiHelper.h>
#include <UI/OpenGL.h>
#include <Util/MinMax.h>
#include <Util/Timer.h>
#include <fstream.h>

namespace SemotusVisum {

unsigned
ZTexRenderer::ztexID = 0;

/**
 * Constructor.
 */
ZTexRenderer::ZTexRenderer() : matricesSet( false ) {
  Log::log( ENTER, "[ZTexRenderer::ZTexRenderer] entered" );
  Log::log( ENTER, "[ZTexRenderer::ZTexRenderer] leaving" );
}

/**
   * Destructor.
   */
ZTexRenderer::~ZTexRenderer() {

}

void writePPM( char * filename, char * image, int w, int h ) {
  FILE * f = fopen( filename, "w" );

  fprintf( f, "P6\n%d %d 255\n", w, h );
  fwrite( image, 1, w*h*3, f );
  fclose(f);
}

/** Renders the image into the screen. */
void 
ZTexRenderer::render( const bool doFPS ) {
  Log::log( ENTER, "[ZTexRenderer::render] entered, thread id = " + mkString((int) pthread_self()) );

  // shouldn't need to make context current here

  // TEST_CODE
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[ZTexRenderer::render] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[ZTexRenderer::render] GL_DEPTH_TEST *not* enabled" );
  
  glEnable(GL_DEPTH_TEST);

  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[ZTexRenderer::render] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[ZTexRenderer::render] GL_DEPTH_TEST *not* enabled" );

  
  //if( glIsEnabled( GL_LIGHTING ) == GL_TRUE )
  //  Log::log( DEBUG, "[ZTexRenderer::render] GL_LIGHTING enabled" );
  //else
  //  Log::log( DEBUG, "[ZTexRenderer::render] GL_LIGHTING *not* enabled" );

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glColor3f(1.0, 1.0, 1.0);
  glDisable(GL_LIGHTING);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

  if( glIsEnabled( GL_LIGHTING ) == GL_TRUE )
    Log::log( DEBUG, "[ZTexRenderer::render] GL_LIGHTING enabled" );
  else
    Log::log( DEBUG, "[ZTexRenderer::render] GL_LIGHTING *not* enabled" );

 
  // /TEST_CODE
  
  WallClockTimer timer;
  timer.start();

  // Set up viewing transforms and such
  preRender();

  // Check GL errors
  int error;
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[ZTexRenderer::render] OpenGL error occurred, error code " + error);
  }

  
  // TEST_CODE

  /*  
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);
    glVertex3f(-5, -5, -5);
    glVertex3f(5, -5, -5);
    glVertex3f(5, 5, -5);
    glVertex3f(-5, 5, -5);
  glEnd();
  */
  
  /*
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_TRIANGLES);
    glVertex3f(-5, -2.5, 0);
    glVertex3f(5, -2.5, 0);
    glVertex3f(0, 2.5, 0);
  glEnd();
  */

  // counter-clockwise
  /*
  float vertices[] = {-5, -2.5, 0,
                      5, -2.5, 0,
                      0, 2.5, 0};
  
  Vector3d normal = mkNormal( vertices, 0, true );
    
  glColor3f(1.0, 1.0, 1.0);
  glNormal3f( normal.x, normal.y, normal.z );
  Log::log( DEBUG, "[ZTexRenderer::render] test normal: (" + mkString(normal.x) + ", " + mkString(normal.y) + ", " + mkString(normal.z) + ")" );
  
  //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  
  glBegin(GL_TRIANGLES);
    glVertex3f(-5, -2.5, 0);
    glVertex3f(5, -2.5, 0);
    glVertex3f(0, 2.5, 0);
  glEnd();
  glEndList();
  */

  glColor3f(1.0, 1.0, 1.0);
  //glutSolidSphere(0.4, 20, 30);
  
  // /TEST_CODE
  
  
  // Now, we render all our ZTex objects
  Log::log( DEBUG, "[ZTexRenderer::render] Drawing " + mkString(displayLists.size()) +
	    " ztex display lists" );
  for ( unsigned i = 0; i < displayLists.size(); i++ ){
    if ( displayLists[i].enabled ){
      cerr << "Display lists enabled" << endl;
      glCallList( displayLists[i].list );
    }
  }
  timer.stop();
  if(timer.time() == 0){
    Log::log( ERROR, "[ZTexRenderer::render] divide by zero" );
    Log::log( LEAVE, "[ZTexRenderer::render] leaving" );
    return;
  }
  fps = 1. / timer.time();
  UserInterface::getHelper()->setFPS( fps );

   // Check GL errors
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[ZTexRenderer::render] OpenGL error occurred, error code " + error);
  }
  Log::log( LEAVE, "[ZTexRenderer::render] leaving" );
}

/** Sets testing geometry */
void 
ZTexRenderer::setTestZTex() {
  Log::log( ENTER, "[ZTexRenderer::setTestZTex] entered, thread id = " + mkString((int) pthread_self()) );

  // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  GLubyte s_checkerTexture[64][64][3];
  float triangle[] = { -5, -2.5, -20,
		       0, 2.5, -20,
                       5, -2.5, -20};
#if 1
  float texCoords[] = { 0, 0.4,
			0.5, 0.9,
			1.0, 0.4 };
  float normals[] = { 0, 0, -1,
		      0, 0, -1,
		      0, 0, -1 };
#endif
  // Make a checkerboard pattern
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++)
    {
      s_checkerTexture[i][j][0] = 255; // R
      s_checkerTexture[i][j][1] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // G
      s_checkerTexture[i][j][2] = 0; // B
    }

#if 0
  // TEST #1
  
  // working!
  
  // Create the texture
  GLuint texture;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
  		  GL_LINEAR_MIPMAP_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB,
	       GL_UNSIGNED_BYTE, s_checkerTexture);
  
  // Build mipmaps
  gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, 64, 64, GL_RGB,
		    GL_UNSIGNED_BYTE, s_checkerTexture);
  
  // Create a display list
  GLuint list = glGenLists(1);
  glNewList(list, GL_COMPILE );
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glBindTexture(GL_TEXTURE_2D, texture);
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_TRIANGLES);
  for ( int i = 0; i < 3; i++ ) {
    glTexCoord2f( texCoords[i*2+0],
		  texCoords[i*2+1]);
    glNormal3f( normals[i*3], normals[i*3 + 1], normals[i*3 + 2] );
    glVertex3f( triangle[i*3], triangle[i*3 + 1], triangle[i*3 + 2] );
  }
  glEnd();
  glDisable(GL_TEXTURE_2D);
  glEndList();

  displayLists.push_back( ztexObj( list, 0, 0 ) );
#else
#if 0
  // TEST #2
  
  // working!
  
  Log::log( DEBUG, "[ZTexRenderer::setTestZTex] drawing basic triangle " );
  // Create a display list
  GLuint list = glGenLists(1);
  glNewList(list, GL_COMPILE );

  glColor3f(1.0, 1.0, 1.0);
  glNormal3f(0.0, 0.0, 1.0);
  glBegin(GL_TRIANGLES);
    glVertex3f(-5, -2.5, 0);
    glVertex3f(5, -2.5, 0);
    glVertex3f(0, 2.5, 0);
  glEnd();
  glEndList();

  displayLists.push_back( ztexObj( list, 0, 0 ) );
#else  
#if 0
  // TEST #3
  
  // Add a ztex object with the checkered texture

  // working!
  
  // check texture data to be sent
  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::setTestZTex] texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(s_checkerTexture[i][j][0]) +
                ", " + mkString(s_checkerTexture[i][j][1]) + ", " + mkString(s_checkerTexture[i][j][2]) + ")" );
    }
  }
  
  // This seems to be the simplest test -- just render a triangle and map a texture onto it
  addZTex( triangle, 3, (char *)s_checkerTexture, 64, 64, true, false );
#else
#if 1
  // TEST #4
  
  // Add a ztex object with the checkered texture

  GLubyte texture[70][70][3];

  // Make a checkerboard pattern
  for (int i = 0; i < 35; i++)
  {  
    for (int j = 0; j < 35; j++)
    {
      texture[i][j][0] = 255; // R
      texture[i][j][1] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // G
      texture[i][j][2] = 0; // B
    }
    for (int j = 35; j < 70; j++)
    {
      texture[i][j][0] = 0; // R
      texture[i][j][1] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // G
      texture[i][j][2] = 255; // B
    }
  }
  for (int i = 35; i < 70; i++)
  {  
    for (int j = 0; j < 35; j++)
    {
      texture[i][j][0] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // R
      texture[i][j][1] = 255; // G
      texture[i][j][2] = 0; // B
    }
    for (int j = 35; j < 70; j++)
    {
      texture[i][j][0] = 0; // R
      texture[i][j][1] = 255; // G
      texture[i][j][2] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // B
    }
  }
  
  // This seems to be the simplest test -- just render a triangle and map a texture onto it
  addZTex( triangle, 3, (char *)texture, 70, 70, true, false );

#else
#if 0
  // TEST #5
  
  char * svmXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<setViewingMethod><method method = \"ZTex Transmission\">okay<eyePoint x = \"2.100000\" y = \"1.600000\" z = \"11.500000\"/><lookAtPoint x = \"0.000000\" y = \"0.000000\" z = \"0.000000\"/><upVector x = \"0.000000\" y = \"1.000000\" z = \"0.000000\"/><perspective far = \"116.77768\" fov = \"60.000000\" near = \"0.1553159\"/></method></setViewingMethod>\001\002\003\004\005";

  Log::log( DEBUG, "[ZTexRenderer::setTestZTex] sending setViewingMethod message" );
  
  NetDispatchManager::getInstance().fireCallback( svmXML,
						  strlen( svmXML ),
						  "server" );
  

  // Do a view frame!
  char * XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size=\"12324\" indexed=\"false\" vertices=\"3\" width=\"64\" height=\"64\" texture=\"4096\">Following</viewFrame>\001\002\003\004\005";
  int triLen = 36; // bytes (3 vertices * 3 coordintes * 4 bytes/float)
  int texLen = 64*64*3;
  int fullSize = triLen+texLen+strlen(XML);
  char * fullData = scinew char[ fullSize ];
  memcpy( fullData, XML, strlen(XML) );

  
  HomogenousConvertHostToNetwork( (void *)(fullData + strlen(XML)),
				  (void *)triangle,
				  FLOAT_TYPE,
				  9 );
  HomogenousConvertHostToNetwork( (void *)(fullData +
					   strlen(XML) +
					   triLen),
				  (void *)s_checkerTexture,
				  CHAR_TYPE,
				  texLen );
  
  Log::log( DEBUG, "[ZTexRenderer::setTestZTex] sending view frame" );
  NetDispatchManager::getInstance().fireCallback( fullData,
						  fullSize,
						  "server" );
  // check texture data that was just sent
  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::setTestZTex] texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(s_checkerTexture[i][j][0]) +
                ", " + mkString(s_checkerTexture[i][j][1]) + ", " + mkString(s_checkerTexture[i][j][2]) + ")" );
    }
  }
  
  delete fullData;
#else
  // TEST #6
  
  GLubyte texture[70][64][3];

  // Make a checkerboard pattern
  for (int i = 0; i < 70; i++)
    for (int j = 0; j < 64; j++)
    {
      texture[i][j][0] = 255; // R
      texture[i][j][1] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // G
      texture[i][j][2] = 0; // B
    }
  
  char * svmXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<setViewingMethod><method method = \"Geometry Transmission\">okay<eyePoint x = \"2.100000\" y = \"1.600000\" z = \"11.500000\"/><lookAtPoint x = \"0.000000\" y = \"0.000000\" z = \"0.000000\"/><upVector x = \"0.000000\" y = \"1.000000\" z = \"0.000000\"/><perspective far = \"116.77768\" fov = \"60.000000\" near = \"0.1553159\"/></method></setViewingMethod>\001\002\003\004\005";
  NetDispatchManager::getInstance().fireCallback( svmXML,
						  strlen( svmXML ),
						  "server" );
  
  
  // Do a view frame!
  char * XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size=\"13476\" indexed=\"false\" vertices=\"3\" width=\"64\" height=\"70\" texture=\"4480\">Following</viewFrame>\001\002\003\004\005";
  int triLen = 36;
  int texLen = 70*64*3;
  int fullSize = triLen+texLen+strlen(XML);
  char * fullData = scinew char[ fullSize ];
  memcpy( fullData, XML, strlen(XML) );
  
  HomogenousConvertHostToNetwork( (void *)(fullData + strlen(XML)),
				  (void *)triangle,
				  FLOAT_TYPE,
				  9 );
  HomogenousConvertHostToNetwork( (void *)(fullData +
					   strlen(XML) +
					   triLen),
				  (void *)texture,
				  CHAR_TYPE,
				  texLen );
  NetDispatchManager::getInstance().fireCallback( fullData,
						  fullSize,
						  "server" );
  delete fullData;
#endif
#endif
#endif
#endif
#endif
  
  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[ZTexRenderer::setTestZTex] leaving" );
}

/** Initializes OpenGL for geometry rendering */
void  
ZTexRenderer::initialize() {
  Log::log( ENTER, "[ZTexRenderer::initialize] entered" );
  // Do our superclass. Can remove this func if no extra init needed

  GeometryRenderer::initialize();

  UserInterface::lock();
  OpenGL::mkContextCurrent();

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
  
  OpenGL::finish();
  UserInterface::unlock();
  // TEST_CODE
  // set test geometry
  setTestZTex();
  // /TEST_CODE
  
  Log::log( LEAVE, "[ZTexRenderer::initialize] leaving" );
}

/** Readies a texture for processing - makes sure that the texture is
 *  power-of-two dimensions, and returns scale factors if needs be.
 *  I the texture does not have power-of-two dimenstions, scales texture
 *  to the next larger resolution that has power-of-two dimensions.
 *
 * @param  width       Width of the texture
 * @param  height      Height of the texture
 * @param  texture     Incoming texture
 * @param  newTex      Outgoing texture
 * @param  widthScale  Scale for x coord of texture coords
 * @param  heightScale Scale for y coord of texture coords
 *
 * @return             True if the texture was modified.
 */
bool  
ZTexRenderer::readyTexture( int &width, int &height, char * texture,
			    char *&newTex, float &widthScale,
			    float &heightScale ) {
  Log::log( ENTER, "[ZTexRenderer::readyTexture] entered" );

  // Check to see if width and height are a power of 2
  int Hpower = 1;
  while ( height > Hpower )
    Hpower *= 2;
  int Wpower = 1;
  while ( width > Wpower )
    Wpower *= 2;
  
  // If not, leave texture unmodified and return false
  if ( Hpower == height && Wpower == width ) {
    //newTex = NULL;
    Log::log( DEBUG, "[ZTexRenderer::readyTexture] texture unmodified" );
    return false;
  }
  
  Log::log( DEBUG, "[ZTexRenderer::readyTexture] WP: " + mkString( Wpower ) +
	    " HP: " + mkString( Hpower ) );
  
  // Find the scales for the new texture
  widthScale = (float)width / (float)Wpower;
  heightScale = (float)height / (float)Hpower;

  Log::log( DEBUG, "[ZTexRenderer::readyTexture] WSCALE: " + mkString( widthScale ) +
	    " HSCALE: " + mkString( heightScale ) );
  int newW = (int)(width / widthScale);
  int newH = (int)(height / heightScale);
  Log::log( DEBUG, "[ZTexRenderer::readyTexture] NW: " + mkString( newW ) +
	    " NH: " + mkString( newH ) );
  // Create the new texture and clear it
  newTex = scinew char[ newW * newH * 3 ]; // RGB
  
  // Copy the old texture into the new.
#if 0
  for ( int i = 0; i < newW*3; i++ )
    for ( int j = 0; j < newH; j++ )
      if ( i < width*3 && j < height )
	newTex[ i + j*newW*3 ] = texture[ i + j*width*3 ];
      else 
	newTex[ i + j*newW*3 ] = 0;
#else
  if (0) { // TMP
    for ( int i = 0; i < newH; i++ )
      for ( int j = 0; j < newW * 3; j+=3 ) {
	newTex[ i*newW*3 + j + 0 ] = (char)(( (float)j / (float)(newW*3) ) * 255);
	newTex[ i*newW*3 + j + 1 ] = 0;
	newTex[ i*newW*3 + j + 2 ] = (char)(255 - ( (float)j / (float)(newW*3) ) * 255);
      }
  }
  else {
    memset( newTex, 255, newW * newH * 3 );
    char * o = texture;
    //char * n = newTex + ( height - 1 )*newW * 3;
    char * n = newTex;
    for ( int i = 0; i < height; i++, o+=width*3, /*n-=newW*3*/ n+=newW*3 )
      memcpy( n, o, width * 3 );
    /*for (int j = 0; j < width*3; j++ )
      if ( *o == 0 )
      *n++ = i % 255; 
      else 
      *n++ = *o++;
      */
  }
#endif
  
  // Set the new width/height
  width = newW;
  height = newH;

  Log::log( DEBUG, "[ZTexRenderer::readyTexture] texture modified" );
  
  Log::log( LEAVE, "[ZTexRenderer::readyTexture] leaving" );
  return true;
}

#define DT
void
ZTexRenderer::addZTex( float *vertices,
		       const int numVertices,
		       char  *texture,
		       const int textureWidth,
		       const int textureHeight,
		       const bool replace,
		       const bool lock ) {
  Log::log( ENTER, "[ZTexRenderer::addZTex] entered, thread id = " + mkString((int) pthread_self()) );
  // Do some error checking
  if ( numVertices % 3 != 0 ) {
    Log::log( ERROR, "[ZTexRenderer::addZTex] Vertex count not a multiple of 3. Got " +
	      mkString( numVertices ) );
    return;
  }
  
  // Set the main graphics window...

  if( lock )
  {  
    UserInterface::lock();
  }
  
  OpenGL::mkContextCurrent();

  // TEST_CODE
  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[ZTexRenderer::addZTex] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[ZTexRenderer::addZTex] GL_DEPTH_TEST *not* enabled" );

  glEnable(GL_DEPTH_TEST);

  if( glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    Log::log( DEBUG, "[ZTexRenderer::addZTex] GL_DEPTH_TEST enabled" );
  else
    Log::log( DEBUG, "[ZTexRenderer::addZTex] GL_DEPTH_TEST *not* enabled" );

  // /TEST_CODE

  // TEST_CODE
  // check texture data to be sent

  /*
  Log::log( DEBUG, "[ZTexRenderer::addZTex] printing texel data" );

  
  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::addZTex] 0 texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(GLubyte (*(texture + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 0*sizeof(GLubyte)))) +
                ", " + mkString(GLubyte (*(texture + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 1*sizeof(GLubyte)))) + ", " + mkString(GLubyte (*(texture + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 2*sizeof(GLubyte)))) + ")" );
    }
  }
  */
  
  // /TEST_CODE

  cerr << "Using following matrices for texture coord calculation:" << endl;
  int i;
  cerr << "Modelview: " << endl;
  for ( i = 0; i < 16; i++ ) {
    cerr << modelview[ i ]  << " ";
    if ( (i+1) % 4 == 0 ) cerr << endl;
  }

  {
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();
    Point3d eyep( zEye );
    Point3d lookat( zAt );
    Vector3d up( zUp );

    cerr << "Eye: " << eyep.toString() << endl;
    cerr << "At: " << lookat.toString() << endl;
    cerr << "Up: " << up.toString() << endl;
    
    gluLookAt(eyep.x, eyep.y, eyep.z,
	      lookat.x, lookat.y, lookat.z,
	      up.x, up.y, up.z);
    cerr << "New modelview: " << endl;

    float m[16];
    glGetFloatv( GL_MODELVIEW_MATRIX, m );
    for ( i = 0; i < 16; i++ ) {
      modelview[i] = m[i];
      cerr << modelview[ i ]  << " ";
      if ( (i+1) % 4 == 0 ) cerr << endl;
    }
  }
  
  cerr << endl << "Projection: " << endl;
  for ( i = 0; i < 16; i++ ) {
    cerr << projMatrix[ i ]  << " ";
    if ( (i+1) % 4 == 0 ) cerr << endl;
  }

  viewPort[0] = viewPort[1] = 0; // ejl 4-20
  viewPort[2] = window_width; // ejl 6-3
  viewPort[3] = window_height; // ejl 6-3 
  cerr << endl << "Viewport: " << endl;
  cerr << viewPort[0] << " " << viewPort[1] << " " << viewPort[2] << " " <<
    viewPort[3] << endl;

  {
    float aspect = (float)viewPort[2] / (float)viewPort[3];
    float top    = near * tan( FOV * M_PI / 360.0 );
    float bottom = -top;
    float left   = bottom * aspect;
    float right  = top * aspect;
    float a0 = 2*near/(right-left);
    float a1 = (right+left)/(right-left);
    float a2 = 2*near/(top-bottom);
    float a3 = (top+bottom)/(top-bottom);
    float a4 = -(far+near)/(far-near);
    float a5 = -2*far*near/(far-near);
    
    float pm[16] =
    //{ a0, 0, a1, 0,
    //  0, a2, a3, 0,
    //  0, 0,  a4, a5,
    //  0, 0,  -1, 0 };

    { a0, 0, 0, 0,
      0, a2, a3, 0,
      a1, 0, a4, -1,
      0, 0,  a5, 0 };
  //    loadProjection( pm );
    cerr << endl << "NEW Projection: " << endl;
    for ( i = 0; i < 16; i++ ) {
      projMatrix[ i ] = pm[ i ];
      cerr << projMatrix[ i ]  << " ";
      if ( (i+1) % 4 == 0 ) cerr << endl;
    }
  }

  
  // Resize the texture, if needs be.
  int texWidth = textureWidth;
  int texHeight = textureHeight;
  float w_scale, h_scale;
  char * theTex = texture;
  if( theTex == NULL )
  {
    Log::log( ERROR, "[ZTexRenderer::addZTex] error copying texture" );
  }    


  // TEST_CODE
  // check texture data to be sent

  /*
  Log::log( DEBUG, "[ZTexRenderer::addZTex] printing texel data" );

  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::addZTex] 1 texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 0*sizeof(GLubyte)))) +
                ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 1*sizeof(GLubyte)))) + ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 2*sizeof(GLubyte)))) + ")" );
    }
  }
  */
  // /TEST_CODE
  
  writePPM("before.ppm", texture, textureWidth, textureHeight );
  writePPM("theTexBefore.ppm", theTex, textureWidth, textureHeight );

  if ( readyTexture( texWidth,
		     texHeight,
		     texture,
		     theTex,
		     w_scale,
		     h_scale ) ) {
    Log::log( MESSAGE, "[ZTexRenderer::addZTex] Rescaling texture from " +
	      mkString( textureWidth ) + " x " + mkString(textureHeight) +
	      " to " +
	      mkString( texWidth ) + " x " + mkString(texHeight) );
    writePPM("after.ppm", texture, texWidth, texHeight );
    writePPM("theTexAfter.ppm", theTex, texWidth, texHeight );
  }
  else {
    w_scale = 1.0;
    h_scale = 1.0;
    Log::log( MESSAGE, "[ZTexRenderer::addZTex] Texture does not need to be rescaled." );
  }
  

  // TEST_CODE
  //w_scale = 1.0;
  //h_scale = 1.0;
  // /TEST_CODE
  
  Log::log( MESSAGE, "[ZTexRenderer::addZTex] Adding " + mkString( numVertices / 3 ) +
	    " triangles, with a texture of size " +
	    mkString( texWidth ) + " x " + mkString(texHeight) );

    
  // The number of floating point numbers we'll be dealing with.
  int numNumbers = numVertices * 3;

  // TEST_CODE
  // check texture data to be sent

  /*
  Log::log( DEBUG, "[ZTexRenderer::addZTex] printing texel data" );

  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::addZTex] 2 texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 0*sizeof(GLubyte)))) +
                ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 1*sizeof(GLubyte)))) + ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 2*sizeof(GLubyte)))) + ")" );
    }
  }
  */
  // /TEST_CODE
  
  // Create the texture
  GLuint textureID;
  
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  cerr << "Binding initial texture " << textureID << " ...";
  UserInterface::getHelper()->checkErrors();
  cerr << "done!" << endl;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
  		  GL_LINEAR_MIPMAP_LINEAR);
  cerr << "GLTexParameter..." << endl;
  UserInterface::getHelper()->checkErrors();
  cerr << "done!" << endl;
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth,
	       texHeight, 0, GL_RGB,
	       GL_UNSIGNED_BYTE, theTex/*texture*/); 
  cerr << "GLTexImage..." << endl;
  UserInterface::getHelper()->checkErrors();
  // Check GL errors
  int error;
  if((error = glGetError()) != GL_NO_ERROR){
    Log::log( ERROR, "[GeometryRenderer::preRender] OpenGL error occurred: " + mkString(gluErrorString(error)) );
  }
  cerr << "done!" << endl;

  // TEST_CODE
  // check texture data to be sent

  /*
  Log::log( DEBUG, "[ZTexRenderer::addZTex] printing texel data" );

  for (int i = 0; i < 64; i++)
  {  
    for (int j = 0; j < 64; j++)
    {
      Log::log( DEBUG, "[ZTexRenderer::addZTex] 3 texel (" + mkString(i) + ", " + mkString(j) + ") : " + "(" + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 0*sizeof(GLubyte)))) +
                ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 1*sizeof(GLubyte)))) + ", " + mkString(GLubyte (*(theTex + i*sizeof(GLubyte)*3 + j*sizeof(GLubyte)*3 + 2*sizeof(GLubyte)))) + ")" );
    }
  }
  */
  // /TEST_CODE
  
  Log::log( DEBUG, "[ZTexRenderer::addZTex] building mipmaps" );
  if ( gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, texWidth,
  			 texHeight, GL_RGB,
  			 GL_UNSIGNED_BYTE,
  			 theTex/*texture*/) != 0 )
    Log::log( ERROR, "[ZTexRenderer::addZTex] GLU BUILD MIPMAP ERROR!" );

  Log::log( DEBUG, "[ZTexRenderer::addZTex] checking for errors" );
  UserInterface::getHelper()->checkErrors();
  cerr << "Did gl tex readying" << endl;

  // Create a display list
  GLuint list = glGenLists(1);
  if ( list != 0 ) {
    glNewList(list, GL_COMPILE );
    Log::log( DEBUG, "[ZTexRenderer::addZTex] executing display list" );
    glEnable(GL_TEXTURE_2D); // TEST_CODE
    UserInterface::getHelper()->checkErrors();
    cerr << "Enabled textures" << endl;
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    cerr << "Texture ID = " << textureID << endl;
    glBindTexture(GL_TEXTURE_2D, textureID);
    UserInterface::getHelper()->checkErrors();
    cerr << "Bound textures" << endl;
    // Set the material properties - TMP
    //GLfloat red[] = {1,0,0,1};
    //glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,    //		  red );
    glBegin(GL_TRIANGLES);
        
    // 3 floats/vertex, 3 vertices/triangle = increment by 9
    float texX[3], texY[3];
    float min[2], max[2];
    min[0] = min[1] = MAXFLOAT;
    max[0] = max[1] = -MAXFLOAT;
    Vector3d normal;
#ifdef DT
    ofstream outfile("CCoords");
#endif
    for ( int i = 0; i < numNumbers; i+=9 ) {
      
      // Tex coords
      mkTexCoords( &vertices[i], texX[0], texY[0] );
      texX[ 0 ] *= w_scale; texY[ 0 ] *= h_scale;
#ifdef DT
      outfile << "Vertex: " << vertices[i] << ", " << vertices[i+1] <<
	", " << vertices[i+2] << " -> Texture: " << texX[0] << ", " <<
	texY[0] << endl;
#endif

      mkTexCoords( &vertices[i+3], texX[1], texY[1] );
      texX[ 1 ] *= w_scale; texY[ 1 ] *= h_scale;
#ifdef DT
      outfile << "Vertex: " << vertices[i+3] << ", " << vertices[i+1+3] <<
      ", " << vertices[i+2+3] << " -> Texture: " << texX[1] << ", " <<
      texY[1] << endl;
#endif
      

      mkTexCoords( &vertices[i+6], texX[2], texY[2] );
      texX[ 2 ] *= w_scale; texY[ 2 ] *= h_scale;
#ifdef DT
      outfile << "Vertex: " << vertices[i+6] << ", " << vertices[i+1+6] <<
      ", " << vertices[i+2+6] << " -> Texture: " << texX[2] << ", " <<
      texY[2] << endl;
#endif
      
      
      
      // Normals
      normal = mkNormal( &vertices[ i ], 0, true );
      glNormal3f( normal.x, normal.y, normal.z );
      Log::log( DEBUG, "[ZTexRenderer::addZTex] normal for this triangle: (" + mkString(normal.x) + ", " + mkString(normal.y) + ", " + mkString(normal.z) + ")" );
      
#ifdef DT
      outfile << "Normal: " << normal.toString() << endl;
#endif

      min[0] = Min( min[0], Min( texX[0], texX[1], texX[2] ) );
      min[1] = Min( min[1], Min( texY[0], texY[1], texY[2] ) );
      max[0] = Max( max[0], Max( texX[0], texX[1], texX[2] ) );
      max[1] = Max( max[1], Max( texY[0], texY[1], texY[2] ) );
      // Combine tex coords, normals, and vertices
      glTexCoord2f( texX[0], texY[0] );
      //glTexCoord2f( 0.0 * w_scale, 0.4 * h_scale ); // TEST_CODE
      glNormal3f( normal.x, normal.y, normal.z );
      glVertex3fv( vertices + i );

      glTexCoord2f( texX[1], texY[1] );
      //glTexCoord2f( 0.5 * w_scale, 0.9 * h_scale ); // TEST_CODE
      glNormal3f( normal.x, normal.y, normal.z );
      glVertex3fv( vertices + i+3 );

      glTexCoord2f( texX[2], texY[2] );
      //glTexCoord2f( 1.0 * w_scale, 0.4 * h_scale ); // TEST_CODE
      glNormal3f( normal.x, normal.y, normal.z );
      glVertex3fv( vertices + i+6 );
    }
    
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
    glEndList();
  
    displayLists.push_back( ztexObj( list, textureID, ztexID++ ) );
    cerr << "Min Tex coords: " << min[0] << ", " << min[1] << endl;
    cerr << "Max Tex coords: " << max[0] << ", " << max[1] << endl;

#ifdef DT
  outfile.close();
#endif
  }
  else {
    Log::log( ERROR, "[ZTexRenderer::addZTex] Unable to allocate display list for geom renderer!" );
  }

  UserInterface::getHelper()->checkErrors();
  OpenGL::finish();

  if( lock )
  {  
    UserInterface::unlock();
  }

  Log::log( LEAVE, "[ZTexRenderer::addZTex] leaving" );
}

/**
 * Sets the current modelview, projection, viewport matrices
 * (for ztex grabs).
 *
 * @param         model         Current modelview matrix (4x4)
 * @param         proj          Current projection matrix (4x4)
 * @param         viewport      Current viewport (1x4)
 */
void   
ZTexRenderer::setMatrices( const double * model,
			   const double * proj,
			   const int * viewport ) {
  Log::log( ENTER, "[ZTexRenderer::setMatrices] entered" );
  memcpy( modelview, model, 16 * sizeof(double) );
  memcpy( projMatrix, proj, 16 * sizeof(double) );
  memcpy( viewPort, viewport, 4 * sizeof(int) );
  matricesSet = true;
  Log::log( LEAVE, "[ZTexRenderer::setMatrices] leaving" );
}


/** Calculates the appropriate texture coordinates for the vertex,
      and stores them in texX and texY */
void  
ZTexRenderer::mkTexCoords( float * vertices,
			   float &texX,
			   float &texY ) {

  // shouldn't need to make context current
  
  //Log::log( ENTER, "[ZTexRenderer::mkTexCoords] entered" );
  GLdouble x,y,z;

  // If matrices aren't set, return.
  //if ( !matricesSet ) {
  //  texX = -1; texY = -1;
  //  return;
  //}

  // Project the vertex into 2D space.
  if ( gluProject( vertices[ 0 ],
		   vertices[ 1 ],
		   vertices[ 2 ],
		   modelview,
		   projMatrix,
		   viewPort,
		   &x,
		   &y,
		   &z ) == GL_FALSE ) {
    texX = -1; texY = -1;
    return;
  }

  texX = x / viewPort[2];
  texY = y / viewPort[3];

  //Log::log( LEAVE, "[ZTexRenderer::mkTexCoords] leaving" );
  // We ignore Z, as we're generating 2D coords.
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
ZTexRenderer::enableGeometryAt( const string ID, const bool enable ) {

}

/**
   * Deletes the geometric object (including annotations) with the
   * given ID.
   *
   * @param ID  ID of the object.
   *
   */
void 
ZTexRenderer::deleteAnnotation( const string ID ) { 
  Log::log( ENTER, "[ZTexRenderer::deleteAnnotation] entered" );

   // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  if ( startsWith( ID, "ZTex" ) ) {
    int ztexID = atoi( ID.substr(4) );
    Log::log( DEBUG, "[ZTexRenderer::deleteAnnotation]Deleting ZTex " + ID +
	      " ID = " + mkString( ztexID ) );
    vector<ztexObj>::iterator i;
    
    for ( i = displayLists.begin(); i != displayLists.end(); i++ )
      if ( i->ID == (unsigned)ztexID ) {
	glDeleteLists( i->list, 1 );
	displayLists.erase(i);

	OpenGL::finish();
        UserInterface::unlock();
        cerr << "End of ZTexRenderer::deleteAnnotation" << endl; 
	return;
      }
  }
  else 
    GeometryRenderer::deleteAnnotation( ID );

  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[ZTexRenderer::deleteAnnotation] leaving" );
}

/**
 * Returns the names (IDs) of all the geometric objects currently in
 * the scene.
 *
 * @return    A list of the IDs of all the geometric objects in the
 *            scene. 
 */
vector<string> 
ZTexRenderer::getGeometryNames() { 
  Log::log( ENTER, "[ZTexRenderer::getGeometryNames] entered" );
  vector<string> names( displayLists.size() );

  for ( unsigned j = 0; j < displayLists.size(); j++ )
    names[j] = string("ZTex ") + mkString( displayLists[j].ID );
  
  Log::log( LEAVE, "[ZTexRenderer::getGeometryNames] leaving" );
  return names;
}

/**
 * Called when we receive another viewing frame from the server.
 * 
 * @param       message           ViewFrame message.
 */
void 
ZTexRenderer::getViewFrame( MessageData * message ) { 
  Log::log( ENTER, "[ZTexRenderer::getViewFrame] entered" );
  ViewFrame *vf = (ViewFrame *)(message->message);
  char * unconvertedData=NULL;
  bool replace = false;
  
  Log::log( DEBUG, "[ZTexRenderer::getViewFrame]  Got a zr view frame!" );
  
  /* Get the original size of the data */
  int origSize = vf->getFrameSize();
  if ( origSize < 0 ) 
    return;

  /*  GET RENDERING DATA */
  if ( vf->isReplaceSet() ) replace = vf->isReplace();
  int numVertices = vf->getVertexCount();
  int numIndices  = vf->getIndexCount();
  int numPolygons = vf->getPolygonCount();
  int textureSize = vf->getTextureDimension();
  int width = vf->getWidth();
  int height = vf->getHeight();
  
  Log::log( DEBUG, "[ZTexRenderer::getViewFrame]  Got " + mkString(numVertices) + " vertices, " +
	    mkString(numIndices) + " indices, and " + mkString(numPolygons) +
	    " polygons, and a texture of size (" 
	    + mkString(width) + ", " + mkString(height) + ") = " +
	    mkString(textureSize) + " bytes.");

  /* If the number of vertices is less than 3, or the number of indices
     (if present) is less than 3, or the texture size is less than 1,
     note that in the log and exit! */
  if ( numVertices < 3 ) {
    Log::log( ERROR, "[ZTexRenderer::getViewFrame] Not enough vertices! Got " + mkString(numVertices) );
    return;
  }
  if ( numIndices != -1 && numIndices < 3 ) {
    Log::log( ERROR, "[ZTexRenderer::getViewFrame] Not enough indices! Got " + mkString(numIndices) );
    return;
  }
  if ( textureSize < 1 ) {
    Log::log( ERROR, "[ZTexRenderer::getViewFrame] Texture too small! Got " + mkString(textureSize) );
    return;
  }

  /* Decompress the data if needs be */
  if ( compressionMode != NONE ) {
    Log::log( DEBUG, string("[ZTexRenderer::getViewFrame] Decompressing viewframe for ") +
	      mkString(message->size) +
	      " bytes. Decompressed length is " + mkString(origSize) );
    unconvertedData = decompressFrame( message->data, 0,
				       message->size, origSize );
    delete message->data; // Because we do not need it anymore
    if ( unconvertedData == NULL ) {
      Log::log( ERROR, "[ZTexRenderer::getViewFrame] Error decompressing image data!" );
      return;
    }
  }
  else {
    // Sanity test. Barf if size != origsize
    if ( message->size != origSize ) {
      Log::log( ERROR, string("[ZTexRenderer::getViewFrame] Uncompressed frame size error! We need ") +
		mkString( origSize ) + ", and got " +
		mkString( message->size ) );
      return;
    }
    Log::log( DEBUG, "[ZTexRenderer::getViewFrame] Using uncompressed data." );
    unconvertedData = message->data;
  }

  /* Now, our data falls into several cases: */
  
  /* Triangles only, not indexed. */
  if ( 1 || numPolygons >= 0 && !vf->isIndexed() ) {
    
    /* Convert the first numVertices as floats; the remainder of the
       data as integers. */
    
    // 3 points/vertex = 3 floats/vertex
    int floatLen = numVertices * 3; // Number of floats
    
    float * floatData = scinew float[ floatLen ];
    
    int charLen = origSize - floatLen*sizeof(float); // Number of chars
    char * charData = scinew char[ charLen ];

    { // Floats
      Log::log( DEBUG, "[ZTexRenderer::getViewFrame] Converting " + mkString( floatLen ) +
		" floats, for a total size of " +
		mkString(floatLen * sizeof(float)) );
      float * undata = (float *)unconvertedData;
      cerr << "Vertex B0: " << undata[0] << endl;
      DataDescriptor d = SIMPLE_DATA( FLOAT_TYPE, 1 );
      for ( int q = 0; q < floatLen; q++ ) {
	ConvertHostToNetwork( (void*)&floatData[q],
			      (void*)&undata[q],
			      &d,
			      1 );
	
      }
      //memcpy( floatData, undata, floatLen );
      cerr << "Vertex 0: " << floatData[0] << endl;
    }

    
    // TMP
    if (0) {
      char * cd = unconvertedData + floatLen * sizeof(float);
      FILE * f = fopen( "texb.out", "w" );
      for ( int i = 0; i < charLen; i+=3 ) {
	fprintf(f, "%c ", cd[i] );
	if ( (i+1) % width == 0 )
	  fprintf( f, "\n" );
      }
      fclose(f);
    }
    // /TMP
    
    { // Chars
      char * cd = unconvertedData + floatLen * sizeof(float);
      Log::log( DEBUG, "[ZTexRenderer::getViewFrame] Converting " + mkString( charLen ) +
		" chars." );
      /* DataDescriptor d = SIMPLE_DATA( CHAR_TYPE, 1 );
	 for ( int q = 0; q < charLen; q++ ) {
	 ConvertHostToNetwork( (void*)&charData[q],
	 (void*)&cd[q],
	 &d,
	 1 );
	 }
      */
      // Otherwise, R and B are swapped! :(
      NetConversion::convertRGB( cd, charLen, charData );
    }

   // TMP
    if (0) {
      char * cd = charData;
      FILE * f = fopen( "tex.out", "w" );
      for ( int i = 0; i < charLen; i+=3 ) {
	fprintf(f, "%c ", cd[i] );
	if ( (i+1) % width == 0 )
	  fprintf( f, "\n" );
      }
      fclose(f);
    }
    // /TMP
    
    
    // Sanity check: #vertices should = 1/3 length of float data
    if ( numVertices != floatLen / 3 ) {
      Log::log( ERROR, 
		"[ZTexRenderer::getViewFrame] Number of vertices present does not equal number claimed: " +
		mkString( floatLen / 3 ) + " / " + mkString(numVertices) );
      
      // Match it to what we have, anyway...
      numVertices = floatLen / 3;
    }

    // Sanity check: rawTex size = width * height.
    if ( width * height * 3 != charLen ) {
      Log::log( ERROR,
	       "[ZTexRenderer::getViewFrame] Texture size does not match number claimed: " +
		mkString(charLen) + " / " +
		mkString(width * height * 3) );
      return;
    }

    // Add the ZTex object

    
    addZTex( floatData,
	     numVertices,
	     charData,
	     width,
	     height,
	     replace,
	     true );
    

  }
  else {
    Log::log( ERROR, "[ZTexRenderer::getViewFrame] Unsupported ztex format!" );
    return;
  }

  // Redraw the screen!
  Log::log( DEBUG, "[ZTexRenderer::getViewFrame] Redrawing after ztex view frame" );

  /* Tell the user interface that we have new data */
  if ( (void *)redraw != NULL )
    redraw();

  Log::log( LEAVE, "[ZTexRenderer::getViewFrame] leaving" );
}


/**
 * Called when we receive an acknowledgement of our change of viewing
 * methods (or viewing parameters) from the server.
 * 
 * @param       message           Message from the server.
 */
void 
ZTexRenderer::setViewingMethod( MessageData * message ) {  
  Log::log( ENTER, "[ZTexRenderer::setViewingMethod] entered" );
  // For right now, use our superclass explicitly
  GeometryRenderer::setViewingMethod( message );

  return;
#if 0
  // We also set our 'global viewing' parameters.
  GLfloat m[16];
  float m1[4][4];
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  gluLookAt( eyePoint.x, eyePoint.y, eyePoint.z,
	     lookAtPoint.x, lookAtPoint.y, lookAtPoint.z,
	     upVector.x, upVector.y, upVector.z );
  
  glGetFloatv( GL_MODELVIEW_MATRIX, m );

  for (int i = 0; i < 4; i++ )
    for (int j = 0; j < 4; j++ )
      m1[i][j] = m[i+4*j];

  Point3d u( upVector.x, upVector.y, upVector.z );
  spaceEye = project( m1, eyePoint );
  spaceAt  = project( m1, lookAtPoint  );
  spaceUp  = (Vector3d)project( m1, u );
  spaceUp.z -= spaceAt.z; // HACK - keep me?
#endif
  /*
    cerr << "Orig  Eye = " << eyePoint.toString() << endl;
    cerr << "Space Eye = " << spaceEye.toString() << endl;
    cerr << "Orig  At = " << lookAtPoint.toString() << endl;
    cerr << "Space At = " << spaceAt.toString() << endl;
    cerr << "Orig  Up = " << upVector.toString() << endl;
    cerr << "Space Up = " << spaceUp.toString() << endl;
  */
  Log::log( LEAVE, "[ZTexRenderer::setViewingMethod] leaving" );
}

/**
 * Turns lighting on/off.
 *
 * @param      lighting       True to enable lighting; false to disable. 
 */
void 
ZTexRenderer::setLighting(const bool lighting) {  
  Log::log( ENTER, "[ZTexRenderer::setLighting] entered" );
  // For right now
  GeometryRenderer::setLighting( lighting );
  Log::log( LEAVE, "[ZTexRenderer::setLighting] leaving" );
}

/**
   * Sets the current shading type for rendering.
   *
   * @param        shadingType     Desired shading type.
   *
   */
void 
ZTexRenderer::setShadingType(const int shadingType) { 
  Log::log( ENTER, "[ZTexRenderer::setShadingType] entered" );
  // For right now
  GeometryRenderer::setShadingType( shadingType );
  Log::log( LEAVE, "[ZTexRenderer::setShadingType] levaing" );
}

void  
ZTexRenderer::loadProjection( float *projection ) {
  Log::log( ENTER, "[ZTexRenderer::loadProjection] entered" );
  // Set the main graphics window...
  cerr << "FIXME = " << __FILE__ << ":" << __LINE__ << endl;//glutUI::getInstance().setMainGFXWindow();

  // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  double fov = 90, nr = 1, fr = 1000;
  double aspect = 1.0;
  
  if ( FOV != NOT_SET ) fov = FOV;
  if ( near != NOT_SET ) nr = near;
  if ( far != NOT_SET ) fr = far;
  cerr << "FIXME = " << __FILE__ << ":" << __LINE__ << endl;//aspect = (double)glutUI::getInstance().getWidth() /
  //(double)glutUI::getInstance().getHeight();

  glMatrixMode( GL_PROJECTION );
  glPushMatrix();
  glLoadIdentity();
  gluPerspective( fov, aspect, nr, fr );
  glGetFloatv( GL_PROJECTION_MATRIX, projection );
  glPopMatrix();

  OpenGL::finish();
  UserInterface::unlock();
  
  Log::log( LEAVE, "[ZTexRenderer::loadProjection] leaving" );
}

/**
 * Enables or disables the annotation with the given ID.
 *
 * @param    ID         ID of the annotation
 * @param    enable     True to enable the annotation; false to disable 
 */
void 
ZTexRenderer::enableAnnotation( const string ID, const bool enable ) {
  Log::log( ENTER, "[ZTexRenderer::enableAnnotation] entered" );
  if ( startsWith( ID, "ZTex" ) ) {
    int ztexID = atoi( ID.substr(4) );
    Log::log( DEBUG, (enable ? "[ZTexRenderer::enableAnnotation]Enabling" : "Disabling") +
	      string(" ZTex ") + ID + " ID = " + mkString( ztexID ) );
    for ( unsigned i = 0; i < displayLists.size(); i++ )
      if ( displayLists[i].ID == (unsigned)ztexID ) {
	displayLists[i].enabled = enable;
        Log::log( LEAVE, "[ZTexRenderer::enableAnnotation] leaving" );
	return;
      }
  }
  else 
    GeometryRenderer::enableAnnotation( ID, enable );
  Log::log( LEAVE, "[ZTexRenderer::enableAnnotation] leaving" );
}

vector<string>
ZTexRenderer::getInfo() {
  Log::log( ENTER, "[ZTexRenderer::getInfo] entered" );
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
  Log::log( LEAVE, "[ZTexRenderer::getInfo] leaving" );
  return info;
}
  

}
