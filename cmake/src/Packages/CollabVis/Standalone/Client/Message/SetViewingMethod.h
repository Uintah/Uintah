/*
 *
 * SetViewingMethod: Message that encapsulates a change in the viewing
 *                   method (or viewing parameters).
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __SET_VIEWING_METHOD_H_
#define __SET_VIEWING_METHOD_H_

#include <Message/MessageBase.h>
#include <Logging/Log.h>
#include <Util/Point.h>

namespace SemotusVisum {

/**
 * This class provides the infrastructure to create, read, and serialize
   a SetViewingMethod message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class SetViewingMethod : public MessageBase {
public:

  /**
   *  Constructor
   *
   */
  SetViewingMethod();

  
  /**
   *  Destructor
   *
   */
  ~SetViewingMethod();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   * Returns true if the server response is okay.
   *
   * @return True if the server responded with an okay.
   */
  inline bool getOkay() const { return isOkay; }
  
  /**
   *  Sets the name and version of the renderer
   *
   * @param name       Name of the renderer
   * @param version    Version of the renderer
   */
  inline void setRenderer( const string name, const string version ) {
    rendererName = name;
    rendererVer  = version;
  }
  
  /**
   *  Returns the renderer name if set.
   *
   * @return Renderer name, or "" if not set.
   */
  inline string getRendererName() { return rendererName; }

  
  /**
   *  Returns the renderer version if set.
   *
   * @return Renderer version, or "" if not set.
   */
  inline string getRendererVersion() { return rendererVer; }

  
  /**
   *  Sets the render group.
   *
   * @param group    Render group.
   */
  inline void   setRenderGroup( const string group ) {
    if ( group.empty() ) return;
    groupName = group;
  }
  
  /**
   *  Returns the render group if set.
   *
   * @return The render group, or "" if not set.
   */
  inline string getRenderGroup() { return groupName; }
  
  /**
   *  Sets the server viewer.
   *
   * @param viewer    Server viewer.
   */
  inline void   setViewer( const  string viewer ) {
    if ( viewer.empty() ) return;
    viewerName = viewer;
  }
  
  /**
   *  Returns the server viewer, if set
   *
   * @return The server viewer, or "" if not set.
   */
  inline string getViewer() { return viewerName; }
  
  /**
   *  Sets the fog parameter.
   *
   * @param fog   True to have fog in the message
   */
  inline void setFog( const bool fog ) {
    isFog = fog ? 1 : 0 ;
  }

  /**
   * Returns true if there is fog info in the message.
   *
   * @return   True if fog info is present. Else false.
   */
  inline bool isFogSet() const {
    if ( isFog == -1 ) return false;
    return true;
  }

  /**
   * Returns true if fog is true.
   *
   * @return   True if fog is true. Else false.
   */
  inline bool getFog() const {
    if ( isFog == 0 ) return false;
    return true;
  }

  /**
   * Turns lighting on/off.
   *
   * @param   lighting     True to turn lighting on, false to turn it off.
   */
  inline void setLighting( const bool lighting ) {
    isLighting = lighting ? 1 : 0;
  }
  
  /**
   * Returns true if there is lighting info in the message.
   *
   * @return   True if lighting info is present. Else false.
   */
  inline bool isLightingSet() const {
    if ( isLighting == -1 ) return false;
    return true;
  }

  /**
   * Returns true if lighting is true.
   *
   * @return   True if lighting is true. Else false.
   */
  inline bool getLighting() const {
    if ( isLighting == 0 ) return false;
    return true;
  }

  /**
   * Toggles the given light on/off.
   * 
   * @param    whichLight   The light to toggle. 
   */
  inline void setLight( const int whichLight ) {
    isLight = whichLight;
  }
  
  /** 
   * Sets the image scale to that given.
   *
   * @param    scale    New image scale;  1:scale.
   */
  inline void setScale( const int scale ) {
    isScale = scale;
  }


  /**
   * Returns true if there is scale info in the message.
   *
   * @return   True if scale info is present. Else false.
   */
  inline bool isScaleSet() const {
    if ( isScale == -1 ) return false;
    return true;
  }

  /**
   * Returns scale.
   *
   * @return   Integer scale reduction.
   */
  inline int getScale() const {
    return isScale;
  }

  /**
   * Sets the 'subimage' parameter
   *
   * @param subimage      True to enable subimaging; else false;
   */
  inline void setSubimage( const bool subimage ) {
    isSubimage = subimage ? 1 : 0;
  }

  /**
   * Returns true if subimaging is set in this message.
   *
   * @return True if subimage parameter in message.
   */
  inline bool isSubimageSet() const {
    return isSubimage != -1;
  }

  /**
   * Returns the value of the subimage parameter
   *
   * @return False if set to 'false' - else true (even if not set).
   */
  inline bool getSubimage() const {
    return isSubimage != 0;
  }
  
  /**
   * Sets the shading type.
   *
   * @param    shadingType   Numeric shading type
   */
  inline void setShading( const int shadingType ) {
    isShading = shadingType;
  }

  /**
   * Returns true if there is shading info in the message.
   *
   * @return   True if shading info is present. Else false.
   */
  inline bool isShadingSet() const {
    if ( isShading == -1 ) return false;
    return true;
  }

  /**
   * Returns shading.
   *
   * @return   Numeric ID of shader.
   */
  inline int getShading() const {
    return isShading;
  }

  /**
   * Sets the eyepoint.
   *
   * @param    eye       New eyepoint
   */
  inline void setEyepoint( const Point3d eye ) {
    eyePoint = Point3d(eye);
  }

  /**
   * Returns the eyepoint if set.
   *
   * @return              The eyepoint if set, or null if not set.
   */
  inline Point3d getEyePoint() const {
    return eyePoint;
  }

  /**
   * Sets the look at point.
   *
   * @param    at       New look at point
   */
  inline void setLookAtPoint( const Point3d at ) {
    lookAtPoint = Point3d( at );
  }

  /**
   * Returns the lookatpoint if set.
   *
   * @return              The lookatpoint if set, or null if not set.
   */
  inline Point3d getLookAtPoint() const {
    return lookAtPoint;
  }

  /**
   * Sets the up Vector.
   *
   * @param    up       New up Vector
   */
  inline void setUpVector( const Vector3d up ) {
    upVector = Vector3d( up );
  }

  /**
   * Returns the upVector if set.
   *
   * @return              The upVector if set, or null if not set.
   */
  inline Vector3d getUpVector() const {
    return upVector;
  }
  
  /**
   * Sets the field of view to the supplied param.
   *
   * @param      fov         New field of view
   */
  inline void setFOV( const double fov ) {
    this->fov = fov;
  }

  /**
   * Gets the current field of view
   *
   * @return         The current field of view, or Double.MIN_VALUE if not set.
   */
  inline double getFOV() const {
    return fov;
  }

  /**
   * Sets the near clipping plane to the supplied param.
   *
   * @param      near         New near clipping plane
   */
  inline void setNear( const double near ) {
    this->near = near;
  }

  /**
   * Gets the current near clipping plane
   *
   * @return         The current near clipping plane, or 
   *                 Double.MIN_VALUE if not set.
   */
  inline double getNear() const {
    return near;
  }

  /**
   * Sets the far clipping plane to the supplied param.
   *
   * @param      far         New far clipping plane
   */
  inline void setFar( const double far ) {
    this->far = far;
  }

  /**
   * Gets the current far clipping plane
   *
   * @return         The current far clipping plane, or 
   *                 Double.MIN_VALUE if not set.
   */
  inline double getFar() const  {
    return far;
  }

  /**
   *  Returns a SetViewingMethod message from the given raw data.
   *
   * @param data   Raw data
   * @return A new message, or NULL on error
   */
  static SetViewingMethod * mkSetViewingMethod( void * data );
  
protected:
  /** Renderer name */
  string   rendererName;

  /** Renderer version */
  string   rendererVer;

  /** Render group name */
  string   groupName;

  /** Server viewer name */
  string   viewerName;

  /** Eyepoint */
  Point3d  eyePoint;

  /** Look at point */
  Point3d  lookAtPoint;

  /** Up vector */
  Vector3d upVector;

  /** Field of view */
  double   fov;

  /** Near plane */
  double   near;

  /** Far plane */
  double   far;

  /** True if the server response is okay */
  bool isOkay;

  /** Does this message contain a fog parameter? */
  int isFog;

  /** Does this message contain a lighting parameter? */
  int isLighting;
  
  /** Does this message contain a light parameter? */
  int isLight;

  /** Does this message contain a scale parameter? */
  int isScale;

  /** Does this message contain a shading parameter? */
  int isShading;

  /** Does this message contain a subimage parameter? */
  int isSubimage;

};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:28  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:13  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.5  2001/06/05 17:44:56  luke
// Multicast basics working
//
// Revision 1.4  2001/05/21 19:19:30  luke
// Added data marker to end of message text output
//
// Revision 1.3  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
