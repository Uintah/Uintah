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
#include <XML/XMLWriter.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   ViewFrame
   
KEYWORDS
   View Frame, Message
   
DESCRIPTION

   This class provides the infrastructure to create and serialize
   a ViewFrame message.
   
****************************************/
class ViewFrame : public MessageBase {
public:

  //////////
  // Constructor.
  ViewFrame();

  //////////
  // Destructor. 
  ~ViewFrame();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Sets the 'size' parameter in the message.
  inline void setSize( unsigned int size ) {
    this->size = size;
  }

  //////////
  // Sets the width and height parameters in the message
  inline void setDimensions( unsigned int width, unsigned int height ) {
    this->width = width; this->height = height;
  }

  inline void setSubimage( const int offX, const int offY,
			   const int fullX, const int fullY,
			   const char bkgd[3] ) {
    this->offX = offX; this->offY = offY;
    this->fullX = fullX; this->fullY = fullY;
    for ( int i = 0; i < 3; i++ )
      this->bkgd[i] = bkgd[i];
  }
  //////////
  // Sets whether or not the geometry following is indexed.
  inline void setIndexed( bool indexed ) {
    isIndexed = ( indexed ? 1 : 0 ); 
  }

  //////////
  // Sets whether or not the geometry following is to replace the current
  // geometry.
  inline void setReplace( bool replace ) {
    isReplace = ( replace ? 1 : 0 );
  }

  //////////
  // Sets the number of vertices.
  inline void setVertices( int vertices ) {
    numVertices = vertices;
  }

  //////////
  // Sets the number of indices.
  inline void setIndices( int indices ) {
    numIndices = indices;
  }

  //////////
  // Sets the number of polygons.
  inline void setPolygons( int polygons ) {
    numPolygons = polygons;
  }

  inline void setTextureSize( int size ) {
    textureSize = size;
  }
  
protected:

  // Size of following viewing info (bytes).
  unsigned int size;

  // Width of the image data.
  int width;

  // Height of the image data.
  int height;

  int offX, offY, fullX, fullY;
  char bkgd[3];
  
  // 1 if indexed, 0 if not, -1 if not set.
  int  isIndexed;

  // 1 if replace, 0 if not, -1 if not set.
  int  isReplace;

  // Number of vertices, -1 if not set.
  int  numVertices;

  // Number of indices, -1 if not set.
  int  numIndices;

  // Number of polygons, -1 if not set.
  int  numPolygons;

  // Texture size, -1 if not set.
  int  textureSize;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
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
