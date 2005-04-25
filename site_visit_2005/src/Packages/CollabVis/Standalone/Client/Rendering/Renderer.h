/*
 *
 * Renderer: Interface for various rendering classes
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#ifndef __Renderer_h_
#define __Renderer_h_

#include <math.h>

#include <Util/stringUtil.h>
#include <Util/Point.h>
#include <Util/Color.h>

#include <Rendering/MouseEvent.h>
#include <Logging/Log.h>
#include <Network/NetDispatchManager.h>
#include <Compression/Compressors.h>
#include <vector>

namespace SemotusVisum {

class MessageData;

/**
 * Enapsulates the concept of a collaboration item (text, pointer, etc.)
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class CollaborationItem {
public:
  /// Pointer type
  static const int POINTER = 0;

  /// Text annotation
  static const int TEXT    = 1;


  /// Drawing annotation
  static const int DRAWING = 2;

  /// Select tolerance for picking, in pixels
  static const double SELECT_TOLERANCE = 5.0;

  
  /**
   * Returns a textual representation of the annotation.
   *
   * @return A textual representation of the annotation.
   */
  inline string toString() {
  
    return string( point.toString() + " Text = " +
		   text + ". color = " + color.toString() + ". type = " +
		   mkString(type) + ". ID = " + ID ) ;
  }
  
  /**
   * Constructor. Initializes type only
   *
   * @param type  Type of annotation
   */
  CollaborationItem( const int type ) : type(type),
					pointerSize(DEFAULT_POINTER_SIZE),
					enabled( true )  {
    cerr << "In  CollaborationItem::CollaborationItem" << endl;
    ID = string( "Local" + mkString(++lastID) );
    cerr << "End of CollaborationItem::CollaborationItem" << endl; 
  }
  
  /**
   *  Constructor. Initializes type and location.
   *
   * @param type   Type of annotation
   * @param point  Location of annotation
   */
  CollaborationItem( const int type, const Point3d point ) :
    type(type), point(point),
    pointerSize(DEFAULT_POINTER_SIZE),
    enabled( true ) {
    cerr << "In CollaborationItem::CollaborationItem" << endl;
    ID = string( "Local" + mkString(++lastID) );
    cerr << "End of CollaborationItem::CollaborationItem" << endl;
  }
  
  /**
   * Constructor. Initializes type, location, and rotation around Z axis.
   *
   * @param type      Type of annotation 
   * @param point     Location of annotation 
   * @param zRotation Rotation around Z axis.   
   */
  CollaborationItem( const int type, const Point3d point,
		     const double zRotation ) : type(type), point(point),
						zRotation(zRotation),
						pointerSize(DEFAULT_POINTER_SIZE),
						enabled( true ) {
    cerr << "In CollaborationItem::CollaborationItem" << endl;
    ID = string( "Local" + mkString(++lastID) );
    cerr << "End of CollaborationItem::CollaborationItem" << endl;
  }
  
  /**
   * Sets x dimension of annotation
   *
   * @param size  Width of annotation
   */
  inline void setXSize( const int size ) {
    cerr << "In setXSize" << endl; 
    xDim = size; 
    cerr << "End of setXSize" << endl; 
  }
    
  /**
   * Sets y dimension of annotation 
   *
   * @param size  Height of annotation  
   */
  inline void setYSize( const int size ) { 
    cerr << "In setYSize" << endl; 
    yDim = size;
    cerr << "End of setYSize" << endl;
  }
  
  /**
   * Returns true if the volume wholly contains the annotation
   *
   * @param lowerX        Lower X boundary
   * @param lowerY        Lower Y boundary
   * @param upperX        Upper X boundary
   * @param upperY        Upper Y boundary
   * @return True if the annotation is wholly contained.
   */
  inline bool contains( const float lowerX, const float lowerY, 
			const float upperX, const float upperY ) {
    cerr << "In contains" << endl;
    cerr << "In contains" << endl;
    return ( point.x >= lowerX && point.x+xDim <= upperX &&
	     point.y >= lowerY && point.y+yDim <= upperY );
  }
  
  /**
   * Returns the type of the annotation
   *
   * @return The type of the annotation
   */
  inline int getType() const { 
    cerr << "In getType" << endl;
    cerr << "End of getType" << endl;
    return type; 
  }
  
  /**
   * Returns the x coord of the annotation
   *
   * @return The x coord of the annotation
   */
  inline float  getX() const { 
    cerr << "In getX" << endl;
    cerr << "End of getX" << endl;
    return point.x; 
  }
  
  /**
   * Returns the y coord of the annotation
   *
   * @return The y coord of the annotation
   */
  inline float  getY() const { return point.y; }

  /**
   * Returns the z coord of the annotation
   *
   * @return The z coord of the annotation
   */
  inline float  getZ() const { return point.z; }
  
  /**
   * Returns the coords of the annotation
   *
   * @return The coords of the annotation
   */
  inline Point3d getPoint() const { return point; }
  
  /**
   * Returns the Z rotation of the annotation
   *
   * @return The Z rotation of the annotation
   */
  inline double getZRotation() const { return zRotation; }
  
  /**
   * Sets the text of text annotations
   *
   * @param text  Text of the annotation
   */
  inline void setText( const string &text ) { this->text = text; }
  
  /**
   * Returns the text of the annotation
   *
   * @return text of the annotation
   */
  inline const string& getText() const { return text; }

  
  /**
   * Returns the ID of the annotation
   *
   * @return The ID of the annotation
   */
  inline const string& getID() const { return ID; }
  
  /**
   * Sets the ID of the annotation
   *
   * @param newID    The new ID of the annotation
   */
  inline void setID( const string &newID ) { ID = newID; }
  
  /**
   * Returns the color of the annotation
   *
   * @return The color of the annotation
   */
  inline Color getColor() const { return color; }
  
  /**
   * Sets the new color of the annotation
   *
   * @param newColor   The new color of the annotation   
   */
  inline  void  setColor( const Color &newColor ) { 
    color = newColor;
  }
  
  /**
   * Adds a point to the drawing annotation
   *
   * @param newPoint      Point to be added
   */
  inline void addSegmentPoint( const Point3d &newPoint ) {
    segments.push_back( newPoint );
  }
  
  /**
   * Returns a list of points in the drawing
   *
   * @return List of points in the drawing
   */
  inline vector<Point3d>& getSegments() { return segments; }
  
  /**
   * Clears the list of points in the drawing
   *
   */
  inline void clearSegments() { segments.clear(); }
  
  /**
   * Sets the line width.
   *
   * @param width  New line width.
   */
  inline void setLineWidth( const int width ) { lineWidth = width; }
  
  /**
   * Returns the line width
   *
   * @return The line width
   */
  inline int getLineWidth() const { return lineWidth; }
  
  /**
   * Sets the text size
   *
   * @param size   New text size  
   */
  inline void setTextSize( const int size ) { textSize = size; }
  
  /**
   * Returns the current text size
   *
   * @return The current text size
   */
  inline int getTextSize() const { return textSize; }
  
  /**
   * Returns the distance between the annotation and the 2D point
   *
   * @param _x    X coord of point
   * @param _y    Y coord of point
   * @return      Straight line distance between annotation and point.
   */
  inline double getBoundsDistance( const float _x, const float _y ) const {
    return sqrt( (point.x-_x)*(point.x-_x) + (point.y-_y)*(point.y-_y) );
  }
  
  /**
   * Sets the size of the pointer.
   *
   * @param size  New pointer size.
   */
  inline void setPointerSize( const float size ) {
    pointerSize = size;
  }
  
  /**
   * Draws the annotation to the screen.
   *
   */
  void draw();
  
  /**
   * Returns whether or not the annotation is enabled.
   *
   * @return True if the annotation is enabled, false if disabled.
   */
  bool isEnabled() const { return enabled; }

  
  /**
   * Enables/Disables annotation
   *
   * @param enable        True to enable, false to disable.
   */
  void enable( const bool enable ) { enabled = enable; } 
  
protected:
  /// Default pointer size
  static const float DEFAULT_POINTER_SIZE = 50.0;

  /// Last annotation ID
  static int lastID;

  /// Annotation type
  int    type;

  /// Location of annotation
  Point3d point;

  /// X dimension of annotation
  int    xDim;

  /// Y dimension of annotation
  int    yDim;

  /// Rotation around Z axis
  double zRotation;

  /// Text, if applicable
  string text;

  /// Pointer color
  Color  color;

  /// ID of the annotation
  string ID;

  /// List of drawing segments
  vector<Point3d> segments;

  /// Line width 
  int    lineWidth;

  /// Text size
  int    textSize;

  /** Draws a string at location (X,Y) */
  void drawString( const string& s, const int x, const int y );

  /// Pointer size
  float  pointerSize;

  /// Enabled status of annotation
  bool   enabled;

};

/**
 * Interface for various rendering classes.
 *
 * @author      Eric Luke
 * @version     $Revision$
 */
class Renderer {
public:
  typedef enum { 
    SHADING_FLAT, /** Flat shading */
    SHADING_GOURAUD, /** Gouraud shading */
    SHADING_WIREFRAME, /** Wireframe shading */
    NUM_SHADERS } shading_t;

  /** Constant for "not set" variables */
  static const double NOT_SET = -MAXFLOAT;
  
  /** Shading strings */
  static const char * const shading[];

  /** Number of lights available for OpenGL lighting */
  static const int NUMBER_OF_LIGHTS        = 6;

  
  /** Version of the renderer */
  static const string        version;
  
  /** Name of the renderer */
  static const string        name;
  
  /** Unique renderer ID */
  static const int           ID      = 0;

  /** Window width */
  int window_width;
  
  /** Window height */
  int window_height;
  
  /** Length of time (in ms) it took to render the current frame */
  long frameTime;

  // Indicates whether or not the viewing parameters have been set by setViewingMethod.
  bool view_set;
  
  /**
   * Returns the name of the renderer.
   *
   * @return Name of the renderer.
   */
  virtual const string Name() { return Renderer::name; }
  
  /**
   * Returns the name of the renderer.
   *
   * @return    The text name of the renderer.
   */
  static inline const string& getName() { return Renderer::name; }

  /**
   * Returns the version of the renderer.
   *
   * @return    The text version of the renderer.
   */
  static inline const string& getVersion() { return Renderer::version; }

  
  /**
   * Sets the eyepoint.
   *
   * @param    eye       New eyepoint
   */
  inline void setEyepoint( const Point3d &eye ) {
    if ( eye.set() )
      eyePoint = eye ;
  }

  /**
   * Returns the eyepoint if set.
   *
   * @return              The eyepoint if set, or null if not set.
   */
  inline Point3d getEyePoint() {
    return eyePoint;
  }
  
  /**
   * Sets the look at point.
   *
   * @param    at       New look at point
   */
   inline void setLookAtPoint( const Point3d &at ) {
    if ( at.set() )
      lookAtPoint = at;
   }
  
  /**
   * Returns the lookatpoint if set.
   *
   * @return              The lookatpoint if set, or null if not set.
   */
  inline Point3d getLookAtPoint() {
    return lookAtPoint;
  }
  
  /**
   * Sets the up Vector.
   *
   * @param    up       New up Vector
   */
  inline void setUpVector( const Vector3d &up ) {
    if ( up.set() )
      upVector = up;
  }
  
  /**
   * Returns the upVector if set.
   *
   * @return              The upVector if set, or null if not set.
   */
  inline Vector3d getUpVector() {
    return upVector;
  }

  /**
   * Sets the perspective info.
   *
   * @param   fov         Field of view
   * @param   nearClip    Near clipping plane
   * @param   farClip     Far clipping plane
   */
  inline void setPerspective( double fov, double nearClip, double farClip ) {
    cerr << "In Renderer::setPerspective" << endl;
    FOV = fov;
    near = nearClip;
    far = farClip;
    cerr << "End of Renderer::setPerspective" << endl;
  }

  /** 
   * Sets the pointer color.
   *
   * @param  newColor    New pointer color.
   */
  inline void setPointerColor( const Color &newColor ) { 
    cerr << "In Renderer::setPointerColor" << endl;
    pointerColor = newColor; 
    Log::log( DEBUG, "Setting new pointer color to " + newColor.toString() );
    cerr << "End of Renderer::setPointerColor" << endl;
  }
  
  /**
   * Returns the pointer color.
   *
   * @return             The current pointer color.
   */
   inline Color getPointerColor() { return pointerColor; }


  /** 
   * Sets the drawing color.
   *
   * @param  newColor    New drawing color.
   */
  inline void setDrawingColor( const Color &newColor ) { 
    drawingColor = newColor; 
    Log::log( DEBUG, "Setting new drawing color to " + newColor.toString() );
  }
  
  /** 
   * Sets the drawing color for the specified drawing.
   *
   * @param  newColor    New drawing color.
   * @param  ID          ID of drawing whose color should be changed.
   */
  inline void setDrawingColor( const Color &newColor, const string &ID ) { 
    CollaborationItem *ci = getID( ID );
    if ( ci == NULL ) return;
    ci->setColor( newColor );
    Log::log( DEBUG, "Setting drawing " + ID + " color to " +
	      newColor.toString() );
  }
  
  
  /**
   * Returns the drawing color.
   *
   * @return             The current drawing color.
   */
  inline Color getDrawingColor() { return drawingColor; }


  /** 
   * Sets the text color.
   *
   * @param  newColor    New text color.
   */
  inline void setTextColor( const Color &newColor ) { 
    textColor = newColor; 
    Log::log( DEBUG, "Setting new drawing color to " + newColor.toString() );
  }
  
  /**
   * Returns the text color.
   *
   * @return             The current text color.
   */
  inline Color getTextColor() { return textColor; }
  
  
  /**
   * Returns true if this renderer allows scaling. 
   */
  virtual inline bool allowScaling() const {
    return scaling;
  }

  /**
   * Constructor for the renderer class. Initializes the network and
   * local user interface managers. Sets our willingness to accept
   * 'viewFrame' and 'setViewingMethod' packets from the server.
   */
  Renderer() : redraw(NULL),fps(-1), compressionMode( NONE ), drawingMode( false ),
	       pointerColor( 255,255,255),
	       textColor( 255, 255, 255 ),
	       drawingColor( 255, 255, 255),
	       lineWidth(1), textSize(2), FOV( NOT_SET ),
	       near( NOT_SET ), far( NOT_SET ) {
    
    vfID = NetDispatchManager::getInstance().registerCallback( VIEW_FRAME,
							       __getViewFrame,
							       this,
							       true );

    svmID =
      NetDispatchManager::getInstance().registerCallback( SET_VIEWING_METHOD,
							  __setViewingMethod,
							  this,
							  true );

    compID =
      NetDispatchManager::getInstance().registerCallback( COMPRESSION,
							  __compression,
							  this,
							  true );
  }

  /**
   * Removes callbacks set in constructor. Should be called before object
   * is deleted.
   */
  virtual ~Renderer() {
    
    NetDispatchManager::getInstance().deleteCallback( vfID );
    NetDispatchManager::getInstance().deleteCallback( svmID );
  }

  /**
   * Sets the new window size.
   *
   * @param         width       Window width
   * @param         height      Window height
   */
  virtual inline void setSize( const int width, const int height ) {
    cerr << "In Renderer::setSize" << endl;
    window_width = width;
    window_height = height;
    cerr << "End of Renderer::setSize" << endl;
  }

  /**
   * Sets the image type to a new type.
   *
   * @param         imageType       Default image type (RGB8, etc.)
   */
  inline void setImageType(const int imageType) {
    cerr << "In Renderer::setImageType" << endl;
    this->imageType = imageType;
    cerr << "End of Renderer::setImageType" << endl;
  }  

  /**
   * Returns the image type.
   *
   * @return  Image type for the renderer.
   */
  inline int getImageType() {
    return imageType;
  }

  /** 
   * Sets the new compression mode based on the given string.
   *
   * @param          mode           String name for compression.
   */
  virtual void setCompressionMode( const string &mode );
  
  /**
   * Sets the new compression mode
   *
   * @param mode     New mode
   */
  virtual inline void setCompressionMode( const int mode ) {
    compressionMode = mode;

  }
  
  /**
   * Compression callback
   *
   * @param obj   Renderer object
   * @param m     Compression message
   */
  static void __compression( void * obj, MessageData *m ) {
    if (!obj) return;
    ((Renderer *)obj)->compression( m );
  }
  
  /**
   * Decompresses the given data, starting at the given offset and 
   * continuing for length bytes.
   *
   * @param         data       Compressed data
   * @param         offset     Offset into data to start from
   * @param         length     Length in bytes of compressed data
   * @param         origSize   Original uncompressed data size.
   *
   * @return                   Newly allocated buffer of uncompressed data,
   *                           or null on an error.
   *
   */
  char * decompressFrame( char * data, const int offset, 
			  const int length, const int origSize );


  /**
   * Rendering method. Called to repaint the screen. 
   *
   * @param         doFPS    Render framerate?
   */
  virtual void render( const bool doFPS=true ) = 0;
  
  /**
   * Receives a mouse event from the rendering window.
   *
   * @param       me           Mouse event.
   */
  virtual void mouseMove( MouseEvent me ) = 0;

  
  /**
   * Callback for view frames
   *
   * @param obj   Renderer object
   * @param m     View frame message
   */
  static void __getViewFrame( void * obj, MessageData *m ) {
    if ( !obj ) return;
    ((Renderer *)obj)->getViewFrame( m );
  }
  
  /**
   * Called when we receive another viewing frame from the server.
   * 
   * @param       message           Message from the server.
   */
  virtual void getViewFrame( MessageData * message ) = 0;


  /**
   * Callback for set viewing method messages
   *
   * @param obj   Renderer object
   * @param m	  Set viewing method message
   */
  static void __setViewingMethod( void * obj, MessageData *m ) {
    if ( !obj ) return;
    ((Renderer *)obj)->setViewingMethod( m );
  }
  
  /**
   * Called when we receive an acknowledgement of our change of viewing
   * methods (or viewing parameters) from the server.
   * 
   * @param       message           Message from the server.
   */
  virtual void setViewingMethod( MessageData *message ) = 0;

  /**
   * Determines the presence of fog. (on/off)
   *
   * @param    show        Whether we want fog.
   */
  virtual void setFog(const bool show) = 0;
  
  /**
   * Turns lighting on/off.
   *
   * @param      lighting       True to enable lighting; false to disable. 
   */
  virtual void setLighting(const bool lighting) = 0;
  
  /**
   * Turns on/off a particular light.
   *
   * @param whichLight      Which light to turn on (0 -> 
   *                        (NUMBER_OF_LIGHTS - 1))
   * @param show            Turn on (true) or off (false).
   *
   * @throws  Exception     If the selected light is invalid.
   */
  virtual void setLight(const int whichLight, const bool show) = 0;
  
  /**
   * Sets image scaling.
   *
   * @param    scale        Scale of image. This is (original size) / 
   *                        (reduced size).
   */
  virtual void setScaling(const int scale) = 0;


  /**
   * Sets the current shading type for image rendering.
   *
   * @param        shadingType     Desired shading type.
   *
   * @throws       Exception       If the shading type is invalid.
   */
  virtual void setShadingType(const int shadingType) = 0;

  /**
   * Sets the new line width for pointer and line annotations.
   *
   * @param        width           New width for the line.
   */
  inline void setLineWidth( const int width ) {
    lineWidth = width;
  }

  /**
   * Returns the line width for pointer and line annotations.
   *
   * @return                       Width for the lines.
   */
  inline int getLineWidth() const { return lineWidth; }

  /**
   * Sets the new text annotation size, in points.
   *
   * @param        size           New size for text.
   */
  inline void setTextSize( const int size ) {
    textSize = size;
    if ( textSize < 1 ) textSize = 1;
  }

  /**
   * Returns the size for text annotations.
   *
   * @return                       Size for text
   */
  int getTextSize() const { return textSize; }

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
   * Adds a pointer at the given location, with the given rotations around
   * the two relevant axes, with the given ID.
   *
   * @param    point        Location of the pointer
   * @param    zRotation    Rotation in radians around the Z axis.
   * @param    ID           ID for this pointer.
   * @param    width        Line width for the pointer.
   * @param    color        Color for the pointer.
   *
   * @return                An ID for this pointer.
   */
  virtual string addPointer( const Point3d point, const double zRotation,
			     string ID,
			     const int width, const Color color );
  /**
   * Removes a pointer with the given ID.
   *
   * @param    ID           ID for the pointer.
   */
  virtual inline void deletePointer( const string &ID ) {
    deleteID( ID );
  }

  /**
   * Removes all pointers from scene.
   */
  virtual inline void deleteAllPointers() {
    deleteAllType( CollaborationItem::POINTER );
  }
  

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
   * Adds the text at the given coordinates with the given ID.
   *
   * @param     point      Location of text
   * @param     text       Text to add.
   * @param     ID         ID for this text.
   * @param     size       Text size.
   * @param     color      Text color.
   *
   * @return               An ID for this text.
   */
  virtual string addText( const Point3d point, const string &text,
			  const string ID, const int size,
			  const Color color );
  
  /**
   * Removes the text with the given ID.
   *
   * @param    ID           ID for the text.
   */
  virtual inline void deleteText( const string &ID ) {
    deleteID( ID );
  }

  /**
   * Removes all text from scene.
   */
  virtual inline void deleteAllText() {
    deleteAllType( CollaborationItem::TEXT );
  }

  /**
   * Creates a new drawing using the current drawing and line width
   * parameters. Returns an ID for the drawing.
   *
   * @return     An ID for the drawing.
   */
  virtual string newDrawing();
  
  /**
   * Creates a new drawing using the current drawing and line width
   * parameters. Returns an ID for the drawing.
   *
   * @param      ID        The ID for the drawing.
   * @param      width     Line width
   * @param      color     Color for lines
   *
   * @return     An ID for the drawing.
   */
  virtual string newDrawing( const string ID, const int width,
			     const Color color );
  
  /**
   * Starts/stops freehand drawing mode. 
   *
   * @param     start       True to start drawing; false to stop drawing.
   * 
   * @return                An ID for the current freehand drawing if we're
   *                        just starting to draw. Else, null.
   */
  virtual string drawMode( const bool start );

  /**
   * Adds another segment to the drawing with the given ID.
   *
   * @param        point        Location of the next line segment.
   * @param        ID           Which drawing to add the segment to.
   */
  virtual void addDrawingSegment( const Point3d point, const string &ID );

  /**
   * Finishes the drawing, and sends it to the server...
   *
   * @param       ID            ID of the drawing
   */
  virtual void finishDrawing( const string &ID );
  
  /** 
   * Clears the drawing with the given ID.
   *
   * @param        ID           ID of the drawing to clear.
   */
  virtual inline void clearDrawing( const string &ID ) {
    CollaborationItem *ci = getID( ID );
    
    if ( ci == NULL ) return;
    ci->clearSegments();
  }

  /**
   * Removes the drawing with the given ID.
   *
   * @param    ID           ID for the drawing.
   */
  virtual inline void deleteDrawing( const string &ID ) {
    deleteID( ID );
  }

  /**
   * Removes all drawings from scene.
   */
   virtual inline void deleteAllDrawings() {
     deleteAllType( CollaborationItem::DRAWING );
  }

  /**
   * Removes all annotations.
   */
  virtual void clearAnnotations() {
    deleteAllPointers();
    deleteAllText();
    deleteAllDrawings();
  }

  /**
   * Deletes the annotation at the given coordinates. Returns the 
   * ID for the deleted annotation.
   *
   * @param         x         X coordinate of annotation.
   * @param         y         Y coordinate of annotation.
   *
   * @return                  ID for annotation, or null if no annotation found
   *                          at the coordinates.
   */
  virtual string deleteAnnotation( const int x, const int y ) {
    CollaborationItem *ci = getAnnotationAt( x, y );
    if ( ci == NULL ) return "";
    
    deleteID( ci->getID() );
    return ci->getID();
  }

  /**
   * Deletes the annotation with the given ID.
   *
   * @param         ID        ID for the annotation
   */
  virtual void deleteAnnotation( const string ID ) {
    string newID = ID.substr(1);
    deleteID( newID );
  }
  
  /**
   * Deletes all objects (pointers, text, freehand drawing) in the given
   * bounding box.
   *
   * @param     lowerX     Lower X coordinate of the box.
   * @param     lowerY     Lower Y coordinate of the box.
   * @param     upperX     Upper X coordinate of the box.
   * @param     upperY     Upper Y coordinate of the box.
   */
  virtual void deleteAllInBox( const int lowerX, const int lowerY, 
			       const int upperX, const int upperY ) {
    
    if ( collaborationItemList.size() == 0 ) return;

    for ( int i = 0; i < (int)collaborationItemList.size(); i++ )
      if ( collaborationItemList[ i ].contains( lowerX,
						lowerY,
						upperX,
						upperY ) )
	cerr << "FIXME - REMOVE ITEM FROM LIST!" << endl;
  }

  /**
   * Enables or disables the annotation with the given ID.
   *
   * @param    ID         ID of the annotation
   * @param    enable     True to enable the annotation; false to disable 
   */
  virtual void enableAnnotation( const string ID, const bool enable ) {
    CollaborationItem *ci = getID( ID.substr(1) );
    if ( ci == NULL ) {
      Log::log( WARNING, "No annotation to enable with ID " + ID.substr(1) );
      return;
    }

    Log::log( DEBUG, (enable ? "Enabling" : "Disabling") +
	      string(" annotation ") + ID );
    ci->enable( enable );
  }

  /**
   * Returns a list of the annotation or other IDs items we may be rendering.
   *
   * @return      List of annotation & other IDs
   */
  virtual vector<string> getInfo() = 0;
  
  /**
   *  Returns a list of the IDs of all the annotations in the scene.
   *
   * @return List of annotation IDs.
   */
  vector<string> getAnnotationIDs();

  void (*redraw)();
  
protected:
  /** Current frames per second */
  double fps;
  
  /** IDs for callbacks */
  int vfID, svmID, compID; 
  
  /** Current display type */
  int           imageType;

  /** Current compression mode */
  int compressionMode;//      = NONE;

  /** Does this renderer allow scaling? */ 
  static const bool scaling = false;

  /** List of collaboration items. */
  vector<CollaborationItem> collaborationItemList;

  /** Are we in drawing mode? */
  bool drawingMode;

  /** Current pointer color */
  Color pointerColor;

  /** Current text color */
  Color textColor;

  /** Current drawing color */
  Color drawingColor;

  /** Current line width */
  int lineWidth;

  /** Current text size */
  int textSize;

  /** Current eyepoint (optional) */
  Point3d eyePoint;

  /** Current lookat point (optional) */
  Point3d lookAtPoint;

  /** Current up vector (optional) */
  Vector3d upVector;

  /** Current field of view (optional) */
  double FOV;

  /** Current near clipping plane (optional) */
  double near;

  /** Current far clipping plane (optional) */
  double far;

  /** Called when we get a 'compression' message from the server */
  void compression( MessageData * message ) {
    Log::log( DEBUG, "Got a compression message!" );

    Compression *c = (Compression *)(message->message);
    if ( c->isCompressionValid() )
      setCompressionMode( c->getCompressionType() );
    else {
      Log::log( ERROR, "Invalid compression mode sent to server" );
    }
  }
  
  /** Deletes all annotations with the given type (pointer, text, etc). */
  virtual void deleteAllType( const int type ) {
    if ( collaborationItemList.size() == 0 ) return;

    for ( int i = 0; i < (int)collaborationItemList.size(); i++ )
      if ( collaborationItemList[i].getType() == type )
	cerr << "FIXME - REMOVE ITEM FROM LIST deleteAllType" << endl;
  }
  
  /** Deletes the annotation with the given ID */
  virtual void deleteID( const string ID ) {
    if ( collaborationItemList.size() == 0 ) return;

    vector<CollaborationItem>::iterator i;
    for ( i = collaborationItemList.begin();
	  i != collaborationItemList.end();
	  i++ )
      if ( i->getID() == ID ) {
	Log::log( DEBUG, "Erasing annotation with ID " + ID );
	collaborationItemList.erase( i );
	return;
      }
	

  }

  /** Returns the annotation at the given coordinates. */
  CollaborationItem * getAnnotationAt( const int x, const int y ) {
    
    // We return the annotation closest to the point (within SELECT_TOLERANCE 
    // pixels), or null if no such annotation exists.
    double minDistance = CollaborationItem::SELECT_TOLERANCE + 1.0; 
    if ( collaborationItemList.size() == 0 ) return NULL;

    double distance = 0;
    CollaborationItem *retval = NULL;
    for ( int i = 0; i < (int)collaborationItemList.size(); i++ ) {
      distance = collaborationItemList[i].getBoundsDistance( x, y );
      if ( distance < minDistance ) {
	retval = &collaborationItemList[i];
	minDistance = distance;
      }
    }
    if ( minDistance <= CollaborationItem::SELECT_TOLERANCE ) 
      return retval;
    return NULL;
  }

  /** Returns the annotation with the given ID */
  CollaborationItem* getID( const string &ID ) {
    if ( collaborationItemList.size() == 0 ) return NULL;

    for ( int i = 0; i < (int)collaborationItemList.size(); i++ ) {
      if ( !strcasecmp( collaborationItemList[i].getID(), ID ) )
	return &collaborationItemList[i];
    }
    return NULL;
  }

  /** Returns the list of collaboration IDs */
  vector<CollaborationItem>& getCIs() {
    return collaborationItemList;
  }

  /** Draws the current framerate. If lighting is true, turns lighting on
      before returning. */
  void drawFPS( const bool lighting=false );
};

}
#endif
