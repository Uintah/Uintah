/*
 *
 * ViewFrame: Message that precedes viewing data.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __VIEW_FRAME_H_
#define __VIEW_FRAME_H_

#include <Message/MessageBase.h>
#include <Logging/Log.h>
#include <Util/Color.h>

namespace SemotusVisum {

/**
 * This class provides the infrastructure to create and serialize
 * a ViewFrame message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class ViewFrame : public MessageBase {
public:

  
  /**
   *  Constructor.
   *
   */
  ViewFrame();

  /**
   *  Destructor.
   *
   */
  ~ViewFrame();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   * Returns the frame size.
   *
   * @return    The (uncompressed) size of the frame.
   */
  inline int getFrameSize() const {
    return frameSize;
  }

  /**
   * Returns the width of the image data.
   *
   * @return    The width of the image data.
   */
  inline int getWidth() const {
    return width;
  }

  /**
   * Returns the height of the image data.
   *
   * @return    The height of the image data.
   */
  inline int getHeight() const {
    return height;
  }

  /**
   * Returns the offset from the origin of a subimage in the X direction.
   *
   * @return Subimage X offset.
   */
  inline int getOffX() const {
    return offX;
  }

  /**
   * Returns the offset from the origin of a subimage in the Y direction.
   *
   * @return Subimage Y offset.
   */
  inline int getOffY() const {
    return offY;
  }

  /**
   * Returns full (nominal) width of a subimage.
   *
   * @return Full subimage width.
   */
  inline int getFullX() const {
    return fullX;
  }

  /**
   * Returns full (nominal) height of a subimage.
   *
   * @return Full subimage height.
   */
  inline int getFullY() const {
    return fullY;
  }

  /**
   * Returns the background color of a subimage
   *
   * @return Subimage background color.
   */
  inline Color getBKGD() const {
    return bkgd;
  }
  
  /**
   * Returns true if the message has geometry index information.
   *
   * @return     True if the message has geometry index information.
   */
  inline bool isIndexSet() const {
    if ( indexed == -1 ) return false;
    return true;
  }
  
  /**
   * Returns true if the following geometry is indexed. 
   *
   * @return     True if the following geometry is indexed. Note that this
   *             only makes sends if isIndexSet() is true.
   */
  inline bool isIndexed() const {
    if ( indexed == 0 ) return false; 
    return true;
  }

  /**
   * Returns true if the message has geometry replacement information.
   *
   * @return     True if the message has geometry replacement information.
   */
  inline bool isReplaceSet() const {
    if ( replace == -1 ) return false;
    return true;
  }
  
  /**
   * Returns true if the following geometry should replace the current geom. 
   *
   * @return     True if the following geometry should replace the current 
   *             geometry. Note that this only makes sends if isReplaceSet() 
   *             is true.
   */
  inline bool isReplace() const {
    if ( replace == 0 ) return false; 
    return true;
  }

  /**
   * Returns the number of vertices in the geometry.
   *
   * @return      Number of vertices in geometry, or -1 if not present.
   */
  inline int getVertexCount() const {
    return vertices;
  }

  /**
   * Returns the number of indices in the geometry.
   *
   * @return      Number of indices in geometry, or -1 if not present.
   */
  inline int getIndexCount() const {
    return indices;
  }

  /**
   * Returns the number of polygons in the geometry.
   *
   * @return      Number of polygons in geometry, or -1 if not present.
   */
  inline int getPolygonCount() const {
    return polygons;
  }

  /**
   * Returns the dimension of the texture.
   *
   * @return      Dimension of the texture, or -1 if not present.
   */
  inline int getTextureDimension() const {
    return textureDimension;
  }

  /**
   * Returns a ViewFrame message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static ViewFrame * mkViewFrame( void * data );
  
protected:
  /** Frame size */
   int frameSize;

  /** Frame width */
   int width;

  /** Frame height */
   int height;

  /// Subimage X offset
  int offX;

  /// Subimage Y offset
  int offY;

  /// Full width of subimage
  int fullX;

  /// Full height of subimage
  int fullY;

  /// Subimage background color
  Color bkgd;
  
  /** Indexed info */
   int indexed;

  /** Replacement info */
   int replace;

  /** Number of vertices */
   int vertices;

  /** Number of indices */
   int indices;

  /** Number of polygons */
   int polygons;


  /** Texture dimension */
   int textureDimension;
  
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:29  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:13  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/08/29 19:58:05  luke
// More work done on ZTex
//
// Revision 1.4  2001/07/16 20:29:29  luke
// Updated messages...
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
