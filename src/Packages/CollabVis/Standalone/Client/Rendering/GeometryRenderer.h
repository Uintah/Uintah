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

#ifndef __Geometry_Renderer_h_
#define __Geometry_Renderer_h_

#include <Rendering/Renderer.h>
#include <Util/BBox.h>
#include <GL/gl.h>

namespace SemotusVisum {

/**
 * Encapsulates info necessary to render a geometric object in OpenGL.
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
struct geomObj {
  /**
   *  Constructor.
   *
   * @param list        GL display list
   * @param c           Color for the object
   * @param ID          ID for the object
   */
  geomObj( const GLuint list, const Color& c, const unsigned ID ) :
    list(list), color(c), ID(ID), enabled(true) {}

  /// Display list
  GLuint list;

  /// Object color
  Color color;

  /// Object ID
  unsigned ID;

  /// Is the object enabled?
  bool enabled;
};

class GeomUI;
class View;

/**
 * Renderer dedicated to 3D geometry rendering. 
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
class GeometryRenderer : public Renderer {
public:
  
  /**
   * Constructor.
   */
  GeometryRenderer();
  
  /**
   * Destructor.
   */
  ~GeometryRenderer();

  
  /**
   * Returns the name of the renderer.
   *
   * @return Name of the renderer.
   */
  virtual const string Name() { return GeometryRenderer::name; }
  
  /**
   * Returns the name of the renderer.
   *
   * @return    The text name of the renderer.
   */
  static const char * getName() { return GeometryRenderer::name; }
  
  /**
   * Returns the version of the renderer.
   *
   * @return    The text version of the renderer.
   */
  static const char * getVersion() { return GeometryRenderer::version; }

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

  /** Initializes OpenGL for geometry rendering */
  void initialize();
  
  /**
   * Receives a mouse event from the rendering window.
   *
   * @param       me           Mouse event.
   */
  void mouseMove( MouseEvent me );

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
  void setTestGeometry();

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
   * Returns the names (IDs) of all the geometric objects currently in
   * the scene.
   *
   * @return    A list of the IDs of all the geometric objects in the
   *            scene. 
   */
  vector<string> getGeometryNames();

  /**
   * Goes to the 'home view' - a view that enables us to look at the center
   * of the object, far enough away that we get all of the geometry in our
   * viewing frustum.
   */
  void homeView();

  /**
   * Returns the number of polygons in the scene.
   *
   * @return                The number of polygons in the scene.
   */
  inline int getPolygonCount() const { return polygonCount; }

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
   * Determines the presence of fog. (on/off)
   *
   * @param    show        Whether we want fog.
   */
  void setFog(const bool show);

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
   */
  void setLight(const int whichLight, const bool show);

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
   * Adds a pointer at the given location, with the given rotations around
   * the two relevant axes.
   *
   * @param    point        Location of pointer.
   * @param    zRotation    Rotation in radians around the Z axis.
   *
   * @return                An ID for this pointer.
   */
  virtual string addPointer( const Point3d point, const double zRotation );

  /**
   * Adds the text at the given coordinates.
   *
   * @param     point      Location of the text
   * @param     text       Text to add.
   *
   * @return               An ID for this text.
   */
  virtual string addText( const Point3d point, const string &text );

  /**
   * Adds another segment to the drawing with the given ID.
   *
   * @param        point        Location of the next line segment.
   * @param        ID           Which drawing to add the segment to.
   */
  virtual void addDrawingSegment( const Point3d point, const string &ID );

  /**
   * Enables or disables the annotation with the given ID.
   *
   * @param    ID         ID of the annotation
   * @param    enable     True to enable the annotation; false to disable 
   */
  virtual void enableAnnotation( const string ID, const bool enable );
  
  /**
   * Returns a list of the annotation or other IDs items we may be rendering.
   *
   * @return      List of annotation & other IDs
   */
  virtual vector<string> getInfo();

  /** Computes new near and far planes for the eye and at points, using
      the current bbox. */
  void compute_depth( const View& v, double &near, double &far );
  
  /** Version of the renderer. */
  static const char * version;

  /** Name of the renderer. */
  static const char * name;

  /** Numeric id of the renderer. */
  static const int           ID            = 1;
  
protected:
  /// Our link to an arcball, which interfaces to the UI
  GeomUI * geomUI;
  
  /// Have the view params changed (ie, from SetViewingMethod message?)
  bool viewParamsChanged;

  /// Should we calculate normals?
  bool calculateNormals;

  /// Shading type
  int shading;

  /// Scene polygon count
  int polygonCount;

  /// Is fog enabled?
  bool fog;

  /// Is lighting on?
  bool lighting;

  /// Lights
  bool lights[ NUMBER_OF_LIGHTS ];
  
  /// Display lists
  vector<geomObj> displayLists;

  /// Static ID for generating per-object IDs
  static unsigned geomID;

  /// Index for current geometry color
  int currentGeomColor;

  /// Global geometry bounding box
  BBox bounding_box;

  ///  Is scaling allowed?
  static const bool scaling = false;

  /// Array of geometry colors
  static Color geomColors[];

  /// Number of unique geometry colors
  static const int numGeomColors = 14;


  /// Array of scene light colors
  static Color lightColors[];

  /// Array of scene light positions
  static Point4d lightPos[];

  /// Number of lights / light colors
  static const int numLightColors = 6;


  /// Specular material for geometry
  static const GLfloat mat_specular[];

  /// Ambient material for geometry
  static const GLfloat mat_ambient[];

  /// Diffuse material for geometry.
  static const GLfloat mat_diffuse[];

  /// Shininess for geometry.
  static const GLfloat mat_shininess = 6.0;

  /** 
   * Calculates normal for the vertices in 'vertices' in a CCW direction.
   * If the flip parameter is true, it flips the normal.
   */
  Vector3d mkNormal( float* vertices, const int offset, const bool flip );

  /** Enables/Disables the geometry with the given ID. */
  void enableGeom( const string ID, const bool enable );

  /** Enables/Disables the pointer with the given ID. */
  void enablePointer( const string ID, const bool enable );

  /** Enables/Disables the text with the given ID. */
  void enableText( const string ID, const bool enable );
  
  /** Enables/Disables the drawing with the given ID. */
  void enableDrawing( const string ID, const bool enable );

  /** Adds the geometry to the scene, optionally replacing the current
   *  geometry. */
  void addGeometry( float *vertices, const int numVertices,
		    const bool replace );

  /** Sets up conditions needed (viewing, etc) for geometry rendering */
  void preRender();
  
  /** Projects the point from screen to world space. */
  Point3d projectPoint( const Point3d point );
};

}
#endif
