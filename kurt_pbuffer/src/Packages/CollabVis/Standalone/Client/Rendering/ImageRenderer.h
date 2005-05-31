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

#ifndef __Image_Renderer_h_
#define __Image_Renderer_h_

#include <Rendering/Renderer.h>
#include <Rendering/MouseEvent.h>
#include <Thread/Mutex.h>

//#define TEX_IMAGE

#ifdef TEX_IMAGE
#include <GL/gl.h>
#endif

namespace SemotusVisum {

/**
 * Renderer dedicated to image rendering.
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
class ImageRenderer : public Renderer {
public:
  
  /** Constructor. Does not set the default image type */
  ImageRenderer();

  /** Constructor. Sets the default image type (RGB8, etc.) 
   *
   * @param         imageType         Default image type (RGB8, etc.)
   */
  ImageRenderer(const int imageType);

  /** Constructor. Sets the default image type and dimensions. 
   *
   * @param         imageType         Default image type
   * @param         width             Initial image width
   * @param         height            Initial image height
   */
  ImageRenderer(const int imageType, const int width, const int height);

  /**
   * Returns the name of the renderer.
   *
   * @return Name of the renderer.
   */
  virtual const string Name() { return ImageRenderer::name; }
    
  /**
   * Returns the name of the renderer.
   *
   * @return    The text name of the renderer.
   */
  static const char * getName() { return ImageRenderer::name; }
  
  /**
   * Returns the version of the renderer.
   *
   * @return    The text version of the renderer.
   */
  static const char * getVersion() { return ImageRenderer::version; }
  
  /**
   * Returns true if this renderer allows scaling.
   *
   * @return  True if the renderer allows scaling
   */
  inline bool allowScaling() const {
    return scaling;
  }

  /**
   * Sets the new window size.
   *
   * @param         width       Window width
   * @param         height      Window height
   */
  void setSize( const int width, const int height );

  /**
   * Rendering method. Called to repaint the screen. 
   *
   * @param         doFPS    Render framerate?
   */
  void render( const bool doFPS=true );

  /**
   * Receives a mouse event from the rendering window.
   *
   * @param       me           Mouse event.
   */
  void mouseMove( MouseEvent me );

  /**
   * Called when we receive another viewing frame from the server.
   * 
   * @param     message    View frame message + data.
   */
  void getViewFrame( MessageData * message );

  /**
   * Called when we receive an acknowledgement of our change of viewing
   * methods (or viewing parameters) from the server.
   * 
   * @param     message    Set Viewing method message
   */
  void setViewingMethod( MessageData *  message );

  /**
   * Determines the presence of fog. (on/off)
   *
   * @param    show        Whether we want fog.
   */
  void setFog(const bool show);

  /**
   * Sets the raster image to that given.
   *
   * @param         image           New raster image.
   */
  void setImage(unsigned char * image);

  /**
   * Turns lighting on/off.
   *
   * @param      lighting       True to enable lighting; false to disable. 
   */
  void setLighting(const bool lighting);

  /**
   * Turns on/off a particular light.
   *
   * @param whichLight      Which light to turn on (0 -> 
   *                        (NUMBER_OF_LIGHTS - 1))
   * @param show            Turn on (true) or off (false).
   *
   */
  void setLight(const int whichLight, const bool show);

  /**
   * Sets image scaling.
   *
   * @param    scale        Scale of image. This is (original size) / 
   *                        (reduced size).
   */
  void setScaling(const int scale);

  /**
   * Sets the current shading type for image rendering.
   *
   * @param        shadingType     Desired shading type.
   *
   */
  void setShadingType(const int shadingType);


  /**
   * Returns a list of the annotation or other IDs items we may be rendering.
   *
   * @return      List of annotation & other IDs
   */
  virtual vector<string> getInfo();
  


  /** For testing purposes only! */
  void printTestImage(unsigned char * image);
  void setTestImage();
  
  /** Current image */
  unsigned char *currentImage;//  = null;

  /** Version of the renderer. */
  static const char * version;

  /** Name of the renderer. */
  static const char * name;

  /** Numeric id of the renderer. */
  static const int           ID            = 0;

protected:
  /// Mutex for protecting image and image characteristics
  Mutex imgMutex;
  
  /// Image offset for subimages
  int imgOffX, imgOffY;

  /// Image size for subimages
  int imgX, imgY;

  /// Is the current image a subimage?
  bool subImage;
  
#ifdef TEX_IMAGE
  GLuint texID;
  GLubyte *textureHolder;
  int texSpanX, texSpanY;
  bool textureHolderValid;
#endif
  
  /** Image scale (scale:1) */
  int scale;

  /** Does this renderer allow scaling? */ 
  const static bool scaling = true;

  /** Decompresses the data into an image (ie, JPG or equivalent) */
  char * decompressImage( unsigned char *data, const int offset,
			  const int length, const int origSize );


  
};

}

#endif // ifdef image_renderer
