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

#ifndef __ZTex_Renderer_h_
#define __ZTex_Renderer_h_

#include <Rendering/GeometryRenderer.h>
#include <UI/GeomUI.h>

namespace SemotusVisum {

/**
 * Encapsulates info necessary to render a ZTex object in OpenGL.
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
struct ztexObj {
  /**
   *  Constructor.
   *
   * @param list        GL display list
   * @param texture     Texture ID
   * @param ID          ID for the object
   */
  ztexObj( const GLuint list, const GLuint texture, const unsigned ID ) :
    list(list), texture(texture), ID(ID), enabled(true) {}

  /// Display list
  GLuint list;

  /// Texture ID
  GLuint texture;

  /// Object ID
  unsigned ID;

  /// Is the object enabled?
  bool enabled;
};

/**
 * Renderer dedicated to ZTex rendering. 
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
class ZTexRenderer : public GeometryRenderer {
public:
  
  /**
   * Constructor.
   */
  ZTexRenderer();
  
  /**
   * Destructor.
   */
  ~ZTexRenderer();

  /**
   * Returns the name of the renderer.
   *
   * @return Name of the renderer.
   */
  virtual const string Name() { return ZTexRenderer::name; }

  /**
   * Returns the name of the renderer.
   *
   * @return    The text name of the renderer.
   */
  static const char * getName() { return ZTexRenderer::name; }

  /**
   * Returns the version of the renderer.
   *
   * @return    The text version of the renderer.
   */
  static const char * getVersion() { return ZTexRenderer::version; }

  /**
   * Returns true if this renderer allows scaling. 
   *
   * @return    True if this renderer allows scaling.
   */
  inline bool allowScaling() {
    return scaling;
  }

  /**
   * Rendering method. Called to repaint the screen. 
   *
   * @param         doFPS    Render framerate?
   */
  void render( const bool doFPS=true );

  /** Initializes OpenGL for ztex rendering */
  void initialize();
  
  /**
   * Sets whether or not we should calculate normals on all <b>future</b>
   * geometry additions.
   *
   * @param   calc              True to calculate normals.
   */
  void setCalculateNormals( const bool calc ) {
    calculateNormals = calc;
  }

  /**
   * Returns whether or not we are calculating normals from geometry.
   *
   * @return                      True if we are calculating normals.
   */
  bool getCalculateNormals() const {
    return calculateNormals;
  }

  /** Sets testing geometry - just a single triangle */
  void setTestZTex();

  /**
   * Enables/Disables the geometric object (including annotations)
   * with the given ID. 
   * 
   * @param     ID          ID of the object.
   * @param     enable      True to enable the object. False to
   *                        disable it.
   */
  void enableGeometryAt( const string ID, const bool enable );

  /**
   * Deletes the geometric object (including annotations) with the
   * given ID.
   *
   * @param ID  ID of the object.
   *
   */
  void deleteAnnotation( const string ID );

  /**
   * Enables or disables the annotation with the given ID.
   *
   * @param    ID         ID of the annotation
   * @param    enable     True to enable the annotation; false to disable 
   */
  virtual void enableAnnotation( const string ID, const bool enable );

  
  /**
   * Returns the names (IDs) of all the geometric objects currently in
   * the scene.
   *
   * @return    A list of the IDs of all the geometric objects in the
   *            scene. 
   */
  vector<string> getGeometryNames();

  /**
   * Called when we receive another viewing frame from the server.
   * 
   * @param       message           ViewFrame message.
   */
  void getViewFrame( MessageData * message );

  /**
   * Called when we receive an acknowledgement of our change of viewing
   * methods (or viewing parameters) from the server.
   * 
   * @param       message           Message from the server.
   */
  void setViewingMethod( MessageData * message );

  /**
   * Turns lighting on/off.
   *
   * @param      lighting       True to enable lighting; false to disable. 
   */
  void setLighting(const bool lighting);

  /**
   * Sets image scaling.
   *
   * @param    scale        Scale of image. This is (original size) / 
   *                        (reduced size).
   */
  void setScaling(const int scale) {} // nop

  /**
   * Sets the current shading type for rendering.
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
  
  /**
   * Sets the current modelview, projection, viewport matrices
   * (for ztex grabs).
   *
   * @param         model         Current modelview matrix (4x4)
   * @param         proj          Current projection matrix (4x4)
   * @param         viewport      Current viewport (1x4)
   */
  void setMatrices( const double * model,
		    const double * proj,
		    const int * viewport );

  /**
   *  Loads the projection matrix into the given parameter.
   *
   * @param projection    Preallocated 4x4 float matrix.
   */
  void loadProjection( float *projection );
  
  /** Version of the renderer. */
  static const char * version;

  /** Name of the renderer. */
  static const char * name;

  /** Numeric id of the renderer. */
  static const int           ID            = 2;

  inline Point3d getEye() {
    Point3d pt = geomUI->getView().eyep();
    zEye = pt;
    return pt;
  }

  inline Point3d getAt() {
    Point3d pt = geomUI->getView().lookat();
    zAt = pt;
    return pt;
  }

  inline Vector3d getUp() {
    Vector3d up = geomUI->getView().up();
    zUp = up;
    return up;
  }
  
protected:
  ///  Static ID for creating ZTex objects
  static unsigned ztexID;

  /// ZTex display lists
  vector<ztexObj> displayLists;

  /// Holder for OpenGL modelview matrix
  double modelview[16];

  /// Holder for OpenGL projection matrix
  double projMatrix[16];

  /// Holder for OpenGL viewport vector
  int    viewPort[4];

  /// Are the matrices set?
  bool matricesSet;

  /// Spatial eyepoint
  Point3d zEye;

  /// Spatial lookAtPoint
  Point3d zAt;

  /// Spatial up vector
  Vector3d zUp;

  // Indicates whether or not the viewing parameters have been set by setViewingMethod.
  //bool view_set;
  
  /** Adds a ztex object with the appropriate parameters */
  void addZTex( float *vertices,
		const int numVertices,
		char  *texture,
		const int textureWidth,
		const int textureHeight,
		const bool replace,
		const bool lock );

  /** Readies a texture for processing - makes sure that the texture is
   *  power-of-two dimensions, and returns scale factors if needs be.
   *
   * @param  width       Width of the texture
   * @param  height      Height of the texture
   * @param  texture     Incoming texture
   * @param  newTex      Outgoing texture (or NULL if not modified)
   * @param  widthScale  Scale for x coord of texture coords
   * @param  heightScale Scale for y coord of texture coords
   *
   * @return             True if the texture was modified.
   */
  bool readyTexture( int &width, int &height, char * texture,
		     char *&newTex, float &widthScale,
		     float &heightScale );
     
  /** Calculates the appropriate texture coordinates for the vertex,
      and stores them in texX and texY */
  void mkTexCoords( float * vertices,
		    float &texX,
		    float &texY );
};

}
#endif
