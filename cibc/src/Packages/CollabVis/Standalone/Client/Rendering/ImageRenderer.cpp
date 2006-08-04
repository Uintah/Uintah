/*
 *
 * ImageRenderer: Renderer dedicated to image rendering.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#include <Rendering/ImageRenderer.h>
#include <Network/NetConversion.h>
#include <Network/NetInterface.h>
#include <Util/Timer.h>
#include <UI/UserInterface.h>
#include <UI/uiHelper.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h> // test code
#include <Logging/Log.h>

extern "C" {
#include <Compression/ppm.h>
}

namespace SemotusVisum {

// Assign static variables
const char * ImageRenderer::version = "1.0";
const char * ImageRenderer::name = "Image Streaming";

/** Constructor. Does not set the default image type */
ImageRenderer::ImageRenderer() : currentImage(NULL),
				 imgMutex("IR Mutex"),
				 subImage(false)
#ifdef TEX_IMAGE
  , texID(0), textureHolder(NULL), texSpanX(0), texSpanY(0),
  textureHolderValid(false)
#endif
{
  Log::log( ENTER, "[ImageRenderer::ImageRenderer] entered" );
  // TEST_CODE
  //view_set = true;
  view_set = false;
  //redraw();
  // /TEST_CODE


  Log::log( LEAVE, "[ImageRenderer::ImageRenderer] leaving" );
}

/** Constructor. Sets the default image type (RGB8, etc.) 
 *
 * @param         imageType         Default image type (RGB8, etc.)
 */
ImageRenderer::ImageRenderer(const int imageType) : currentImage(NULL),
						    imgMutex("IR Mutex"),
						    subImage(false)
#ifdef TEX_IMAGE
  , texID(0), textureHolder(NULL), texSpanX(0), texSpanY(0),
  textureHolderValid(false)
#endif
{
  Log::log( ENTER, "[ImageRenderer::ImageRenderer] entered" );
  setImageType(imageType);
  // TEST_CODE
  //view_set = true;
  view_set = false;
  //redraw();
  // /TEST_CODE
  Log::log( LEAVE, "[ImageRenderer::ImageRenderer] leaving" );
}  

  /** Constructor. Sets the default image type and dimensions. 
   *
   * @param         imageType         Default image type
   * @param         width             Initial image width
   * @param         height            Initial image height
   */
ImageRenderer::ImageRenderer(const int imageType, const int width,
			     const int height) : currentImage(NULL),
						 imgMutex("IR Mutex"),
						 subImage(false)
#ifdef TEX_IMAGE
  , texID(0), textureHolder(NULL), texSpanX(0), texSpanY(0),
  textureHolderValid(false)
#endif
{
  Log::log( ENTER, "[ImageRenderer::ImageRenderer] entered" );
  setImageType(imageType);
  // TEST_CODE
  setSize( width, height );
  //view_set = true;
  view_set = false;
  //redraw();
  // TEST_CODE
  Log::log( LEAVE, "[ImageRenderer::ImageRenderer] leaving" );
}  

/**
 * Sets the new window size.
 *
 * @param         width       Window width
 * @param         height      Window height
 */
void
ImageRenderer::setSize( const int width, const int height ) {
  Log::log( ENTER, "[ImageRenderer::setSize] entered" );

  Renderer::setSize( width, height );
  

#ifdef TEX_IMAGE
  cerr << "Processing TEX_IMAGE" << endl;

  // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  texSpanX = 1;
  while ( texSpanX < width )
    texSpanX *= 2;

  texSpanY = 1;
  while ( texSpanY < height )
    texSpanY *= 2;

  if ( textureHolder ) {
    glDeleteTextures( 1, &texID );
    delete textureHolder;
  }
  
  textureHolder = scinew GLubyte[ texSpanX * texSpanY * 3 ];
  // TMP
  memset( textureHolder, 255, texSpanX * texSpanY * 3 );
  
  textureHolderValid = true;

  glutUI::getInstance().setMainGFXWindow();
  glGenTextures(1, &texID);
  glBindTexture(GL_TEXTURE_2D, texID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSpanX,
	       texSpanY, 0, GL_RGB,
	       GL_UNSIGNED_BYTE, textureHolder );
  
  glutUI::getInstance().checkErrors( __HERE__ );

  OpenGL::finish();
  UserInterface::unlock();
  
#else
  if ( currentImage ) delete currentImage;
  currentImage = scinew (unsigned char)[ width * height * sizeof(int) ];
#endif
  Log::log( LEAVE, "[ImageRenderer::setSize] leaving" );
}

/** Renders the image into the screen. 
 *
 * @param         g               Graphics context to draw into.
 */
void 
ImageRenderer::render( const bool doFPS ) {
  Log::log( ENTER, "[ImageRenderer::render] entered" );

  // don't need to make the context current
  
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  WallClockTimer timer;
  timer.start();
   
  if ( currentImage == NULL ) {
    cerr << "No image yet!" << endl;
    return;
  }
  
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, window_width, 0.0, window_height );
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();
  
  // Redraw the screen
  PreciseTimer p;
  p.start();

  //setTestImage(); // test code
  printTestImage(currentImage); // test code

  imgMutex.lock();
  
  if ( subImage ) {
    Log::log( DEBUG, "[ImageRenderer::render] rendering subimage" );
    cerr << "SUBIMAGE" << endl;
    Log::log( DEBUG, "[ImageRenderer::render] imgOffX = " + mkString(imgOffX) + "imgOffY = " + mkString(imgOffY) );
    glRasterPos2i( imgOffX, imgOffY );
    glDrawBuffer( GL_BACK );
    glDrawPixels( imgX, imgY, GL_RGB, GL_UNSIGNED_BYTE, currentImage);
  }
  
  else {
    Log::log( DEBUG, "[ImageRenderer::render] rendering regular image" );
    glRasterPos2i( 0, 0 );

    // TEST_CODE
    GLfloat rasterPosition[4];
    // get current raster position
    glGetFloatv( GL_CURRENT_RASTER_POSITION, rasterPosition );
    Log::log( DEBUG, "[ImageRenderer::render] current raster position: x = " + mkString(rasterPosition[0]) + ", y = " + mkString(rasterPosition[1]) );
    // /TEST_CODE
    
    //glDrawBuffer( GL_BACK );
    glDrawPixels( window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE,
    	  currentImage);

    // Check GL errors
    int error;
    if((error = glGetError()) != GL_NO_ERROR){
      Log::log( ERROR, "[ImageRenderer::render] OpenGL error occurred: " + mkString(gluErrorString(error)) );
    }
  }
  

  imgMutex.unlock();
  
  glFinish();
  
  cerr << "DrawPixels time: " << p.time() * 1000.0 << " ms" << endl;
  p.stop();
  
  
  // Now draw any annotations that we have. 
  Log::log( DEBUG, "Drawing " + mkString(collaborationItemList.size()) +
	    " annotations" );

  glViewport( 0, 0, window_width, window_height );
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, window_width, 0.0, window_height );
  
  for ( unsigned i = 0; i < collaborationItemList.size(); i++ ) {
    //cerr << "In ImageRenderer::render, calling draw, thread id is " << pthread_self() << endl;
    collaborationItemList[i].draw();
    UserInterface::getHelper()->checkErrors();
  }
  
  timer.stop();
  fps = 1. / timer.time();
  UserInterface::getHelper()->setFPS( fps );

  
  Log::log( LEAVE, "[ImageRenderer::render] leaving" );
}  

/**
 * Receives a mouse event from the rendering window.
 *
 * @param       me           Mouse event.
 */
void 
ImageRenderer::mouseMove( MouseEvent me ) {
  Log::log( ENTER, "[ImageRenderer::mouseMove] entered" );
  //cerr << "IR : got a mouse event!" << endl;

  MouseMove *m = scinew MouseMove();
  m->setMove( me.x, me.y, me.button, me.action );
  m->finish();

  NetInterface::getInstance().sendDataToServer( m );

  Log::log( LEAVE, "[ImageRenderer::mouseMove] leaving" );
}

/** Decompresses the data into an image (ie, JPG or equivalent) */
char *
ImageRenderer::decompressImage( unsigned char * data, const int offset,
				const int length, const int origSize ) {
  Log::log( ENTER, "[ImageRenderer::decompress] entered" );
  /* FIXME - fill this in */
  Log::log( LEAVE, "[ImageRenderer::decompress] leaving" ); 
  return NULL;
}

/**
 * Called when we receive another viewing frame from the server.
 * 
 * @param       xmlLength         Length of the XML data.
 * @param       data              Raw data from the server.
 */
void  
ImageRenderer::getViewFrame( MessageData * message ) {
  Log::log( ENTER, "[ImageRenderer::getViewFrame] entered, thread id = " + mkString( (int) pthread_self() ) );

  //cerr << "printing message->data 0 ***********************" << endl;
  //printTestImage((unsigned char *) message->data);

  
  ViewFrame *vf = (ViewFrame *)(message->message);
  unsigned char * unconvertedData=NULL;
  int width=-1;
  int height=-1;

  PreciseTimer timer;
  timer.start();
  
  Log::log( DEBUG, "Got view frame!" );
  
  /* Get the original size of the image */
  int origSize = vf->getFrameSize();
  if ( origSize < 0 ) 
    return;

  // Get the image width
  width = vf->getWidth();
  if ( width == -1 )
    width = window_width;

  // Get the image height
  height = vf->getHeight();
  if ( height == -1 )
    height = window_height;

  //cerr << "printing message->data 1 ***********************" << endl;
  //printTestImage((unsigned char *) message->data);

  // If this is a subimage, glean those params from the message
  int fullX=0, fullY=0;
  int offX=0, offY=0;
  bool isSubimage = false;
  Color bkgd;
  if ( vf->getFullX() != -1 && vf->getFullY() != -1 &&
       vf->getOffX() != -1 && vf->getOffY() != -1 ) {
    isSubimage = true;
    fullX = vf->getFullX();
    fullY = vf->getFullY();
    offX  = vf->getOffX();
    offY  = vf->getOffY();
    bkgd  = vf->getBKGD();
  }
  else
    isSubimage = false;
  
  Log::log( DEBUG, string("Frame is (") + mkString(width) + " x " +
	    mkString(height) + ") ");


  imgMutex.lock();
  if ( isSubimage ) {
    imgOffX = offX;
    imgOffY = offY;
    imgX    = width;
    imgY    = height;
    subImage = true;
    Log::log( DEBUG, "Subimage full size is (" + mkString(fullX) + " x " +
	      mkString(fullY) + "). Subimage offset is (" + mkString(offX) +
	      " x " +  mkString(offY) + "). Background is " +
	      bkgd.toString() );

    
  }
  else
    subImage = false;
  
  /* We restrict the maximum frame size here to avoid making our box
     unusable on an error. Max size is (1600x1200) */
  if ( isSubimage ) {
    if ( fullX <= 1600 && fullY <= 1200 ) {
      if ( fullX != window_width || fullY != window_height ) { 
	//EJLglutUI::getInstance().setDimensions( fullX, fullY );
#ifdef TEX_IMAGE
	setSize( fullX, fullY );
#else
	window_width = fullX;
	window_height = fullY;
#endif
      }
    }
  }
  else 
    if ( width <= 1600 && height <= 1200 ) {
      if ( width != window_width || height != window_height ) { 
	//EJLglutUI::getInstance().setDimensions( width, height );
#ifdef TEX_IMAGE
	setSize( width, height );
#else
	window_width = width;
	window_height = height;
#endif
      }
    }
  imgMutex.unlock();
  
  // If we need to decode the data (ie, it is compressed), do that. 
  if ( compressionMode != NONE ) {
    Log::log( DEBUG, string("Decompressing viewframe for ") +
	      mkString(message->size) +
	      " bytes. Decompressed length is " + mkString(origSize) );
    unconvertedData = (unsigned char *) decompressFrame(message->data, 0,
				       message->size, origSize );
    delete message->data; // Because we do not need it anymore
    if ( unconvertedData == NULL ) {
      Log::log( ERROR, "Error decompressing image data!" );
      return;
    }
  }
  else {
    // Sanity test. Barf if size != origsize
    if ( message->size != origSize ) {
      Log::log( ERROR, string("Uncompressed frame size error! We need ") +
		mkString( origSize ) + ", and got " +
		mkString( message->size ) );
      return;
    }
    unconvertedData = (unsigned char *) message->data;
  }

  //cerr << "printing message->data 2 ***********************" << endl;
  //printTestImage((unsigned char *) message->data);
  
  //cerr << "printing unconvertedData 1 ***********************" << endl;
  //printTestImage(unconvertedData);

  // Convert RGB
  bool doConvert = true;
  if ( compressionMode != NONE ) {
    int type;
    Compressor * c = mkCompressor( compressionMode, type );
    if ( c && (c->needsRGBConvert() == false) )
      doConvert = false;
    delete c;
  }

  if ( doConvert ) {
    unsigned char * newdata = scinew (unsigned char)[ width*height*3];
    NetConversion::convertRGB( (char *)unconvertedData, width*height*3,
			       (char *)newdata );
    delete[] unconvertedData;
    unconvertedData = newdata;
  }

  
  
  // FIXME - Convert data to proper endian-ness, or let OpenGL care for that 
  imgMutex.lock();
  //cerr << "printing unconvertedData 2 ***********************" << endl;
  //printTestImage(unconvertedData);
  setImage( unconvertedData );
  //cerr << "printing currentImage ***********************" << endl;
  //printTestImage(currentImage);
  
  imgMutex.unlock();

  // TEST_CODE
  view_set = true;
  // /TEST_CODE
  
  /* Tell the user interface that we have new data */
  if ( (void *)redraw != NULL )
  //cerr << "In ImageRenderer::getViewFrame, calling redraw, thread id is " << pthread_self() << endl;
    redraw();
  
  timer.stop();
  fps = 1. / timer.time();
  Log::log( LEAVE, "[ImageRenderer::getViewFrame] leaving" );
}

/**
 * Called when we receive an acknowledgement of our change of viewing
 * methods (or viewing parameters) from the server.
 * 
 * @param       xmlLength         Length of the XML data.
 * @param       data              Raw data from the server.
 */
void   
ImageRenderer::setViewingMethod( MessageData *  message ) {
  Log::log( ENTER, "[ImageRenderer::setViewingMethod] entered" );
  /* We don't need to do anything here - the server does all the rendering. */
  Log::log( LEAVE, "[ImageRenderer::setViewingMethod] leaving" );
} 
 
/**
 * Determines the presence of fog. (on/off)
 *
 * @param    mode        Whether we want fog.
 */
void    
ImageRenderer::setFog(const bool show) {
  Log::log( ENTER, "[ImageRenderer::setFog] entered" );
  SetViewingMethod *svm = scinew SetViewingMethod();
  
  svm->setFog( show );
  svm->setRenderer( name, version );
  
  svm->finish();
  
  NetInterface::getInstance().sendDataToServer( svm );
  Log::log( LEAVE, "[ImageRenderer::setFog] leaving" ); 
}

/**
 * Sets the raster image to that given.
 *
 * @param         image           New raster image.
 */
void    
ImageRenderer::setImage(unsigned char * image) {
  Log::log( ENTER, "[ImageRenderer::setImage] entered" );
  
#ifdef TEX_IMAGE

  // make the context current for this thread
  UserInterface::lock();
  OpenGL::mkContextCurrent();
  
  glutUI::getInstance().lock( __HERE__ );
  glutUI::getInstance().checkErrors( __HERE__ );
  
  glutUI::getInstance().setMainGFXWindow();
  glEnable( GL_TEXTURE_2D );
  glBindTexture(GL_TEXTURE_2D, texID);
  glTexSubImage2D( GL_TEXTURE_2D, 0,
		   0, 0,
		   window_width, window_height,
		   GL_RGB, GL_UNSIGNED_BYTE, image );
  glDisable( GL_TEXTURE_2D );
  glutUI::getInstance().checkErrors( __HERE__ );
  glutUI::getInstance().unlock( __HERE__ );

  OpenGL::finish();
  UserInterface::unlock();
  
#else
  
  // If we don't have an image yet, create one.
  if (currentImage != NULL)
    delete currentImage;
  
  currentImage = image;

#endif
  Log::log( LEAVE, "[ImageRenderer::setImage] leaving" );
} 
 
/**
 * Turns lighting on/off.
 *
 * @param      lighting       True to enable lighting; false to disable. 
 */
void    
ImageRenderer::setLighting(const bool lighting) {
  Log::log( ENTER, "[ImageRenderer::setLighting] entered" );
  SetViewingMethod *svm = scinew SetViewingMethod();
  
  svm->setLighting( lighting );
  svm->setRenderer( name, version );
  svm->finish();
  
  NetInterface::getInstance().sendDataToServer( svm );
  Log::log( LEAVE, "[ImageRenderer::setLighting] leaving" );
}    

/**
 * Turns on/off a particular light.
 *
 * @param whichLight      Which light to turn on (0 -> 
 *                        (NUMBER_OF_LIGHTS - 1))
 * @param show            Turn on (true) or off (false).
 *
 * @throws  Exception     If the selected light is invalid.
 */
void    
ImageRenderer::setLight(const int whichLight, const bool show) {
  Log::log( ENTER, "[ImageRenderer::setLight] entered" );
  /* We do nothing here, as the number of lights is currently only
     one on SCIRun. */
  Log::log( LEAVE, "[ImageRenderer::setLight] leaving" );
}  

/**
 * Sets image scaling.
 *
 * @param    scale        Scale of image. This is (original size) / 
 *                        (reduced size).
 */
void    
ImageRenderer::setScaling(int scale) {
  Log::log( ENTER, "[ImageRenderer::setScaling] entered" );
  SetViewingMethod *svm = scinew SetViewingMethod();
  
  svm->setScale( scale );
  svm->setRenderer( name, version );
  svm->finish();
  NetInterface::getInstance().sendDataToServer( svm );
  
  this->scale = scale; // FIXME - wait for a response...
  Log::log( LEAVE, "[ImageRenderer::setScaling] leaving" );
}  
  
/**
 * Sets the current shading type for image rendering.
 *
 * @param        shadingType     Desired shading type.
 *
 * @throws       Exception       If the shading type is invalid.
 */
void    
ImageRenderer::setShadingType(const int shadingType) {
  Log::log( ENTER,"[ImageRenderer::setShadingType] entered" );
  SetViewingMethod *svm = NULL;
  
  /* Validate shading method */
  switch ( shadingType ) {
  case SHADING_FLAT:
  case SHADING_GOURAUD:
  case SHADING_WIREFRAME:
    
    svm = scinew SetViewingMethod();
    
    svm->setShading( shadingType );
    svm->setRenderer( name, version );
    svm->finish();
    NetInterface::getInstance().sendDataToServer( svm );
    
    break;
    
  default:
    
    Log::log( ERROR, "Bad shading method!" );
  }
  Log::log( LEAVE, "[ImageRenderer::setShadingType] leaving" );
}    

void ImageRenderer::printTestImage(unsigned char * image) {
  Log::log( ENTER, "[ImageRenderer::printTestImage] entered" );
  Log::log( DEBUG, "WINDOW_WIDTH = " + mkString(window_width) + ", WINDOW_HEIGHT = " + mkString(window_height) );
  int numColoredPixels = 0;
  for ( int i = 0; i < window_width * window_height * 3; i+=3 ) {
    if((unsigned int)image[ i ] != 0 || (unsigned int)image[ i+1 ] != 0 || (unsigned int)image[ i+2 ] != 0){
      //Log::log( DEBUG, "[ImageRenderer::printTestImage] <" + mkString((unsigned int)image[ i ]) + ", " + mkString((unsigned int)image[ i+1 ]) + ", " + mkString((unsigned int)image[ i+2 ]) + ">  " );
      numColoredPixels++;
    }
  }
  Log::log( DEBUG, "[ImageRenderer::printTestImage] NUM COLORED PIXELS = " + mkString(numColoredPixels) );
  Log::log( LEAVE, "[ImageRenderer::printTestImage] leaving" );
}


void    
ImageRenderer::setTestImage() {
  Log::log( ENTER, "[ImageRenderer::setTestImage] entered" );
  unsigned char * image = new (unsigned char)[ window_width * window_height * 3 ];

  for ( int i = 0; i < window_width * window_height * 3; i+=3 ) {
    //image[ i ] = i % 255;
    //image[ i+1 ] = i % 75;
    //image[ i+2 ] = i % 125;
    image[ i ] = (unsigned char) 0;
    image[ i+1 ] = (unsigned char) 0;
    image[ i+2 ] = (unsigned char) 255;
  }

  setImage( image );
  Log::log( LEAVE, "[ImageRenderer::setTestImage] leaving" );
}

/**
 * Returns a list of the annotation or other IDs items we may be rendering.
 *
 * @return      List of annotation & other IDs
 */
vector<string>     
ImageRenderer::getInfo() {
  Log::log( ENTER, "[ImageRenderer::getInfo] entered" );
  return getAnnotationIDs(); // We have only annotations
  Log::log( LEAVE, "[ImageRenderer::getInfo] leaving" );
}

}
