/*
 *
 * GeometryRenderer: Provides geometry transmission capability.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */


#ifndef __GeometryRenderer_h_
#define __GeometryRenderer_h_

#include <Rendering/Renderer.h>
//#include <Network/AppleSeeds/formatseed.h> // This sucks...
#include <Network/NetConversion.h>
#include <Thread/Mailbox.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {
namespace Rendering {

using namespace Network;

/**************************************
 
CLASS
   GeometryRenderer
   
KEYWORDS
   Rendering, Geometry
   
DESCRIPTION

   This class provides an infrastructure for geometry transmission.
   It provides methods for transmitting and compressing geometry. 
   
****************************************/

class GeometryRenderer : public Renderer {

public:

  //////////
  // Constructor. Allocates all needed memory.
  GeometryRenderer();

  //////////
  // Destructor. Deallocates memory.
  virtual ~GeometryRenderer();

  ////////
  // Adds callbacks to the network dispatch manager. Should be called
  // before the renderer is used.
  void setCallbacks();
  
  ////////
  // Removes callbacks from the network dispatch manager.
  void removeCallbacks();
  
  //////////
  // Transmits the given data to the clients.
  void        sendRenderData( const char * data, int numBytes,
			      bool copy=true );

  //////////
  // Sets the number of vertices in the coming data.
  inline void setVertexCount( const int count ) { vertexCount = count; }

  //////////
  // Sets the number of indices in the coming data.
  inline void setIndexCount ( const int count ) { indexCount = count; }

  //////////
  // Sets the number of polygons in the coming data.
  inline void setPolyCount  ( const int count )  { polygonCount = count; }

  //////////
  // Sets whether or not the coming data is indexed.
  inline void setIndexed    ( bool indexed ) {
    this->indexed = ( indexed ? 1 : 0 );
  }

  
  //////////
  // Sets whether or not the coming data should replace (or be appended to)
  // the current geometry.
  inline void setReplace    ( bool replace ) {
    this->replace = ( replace ? 1 : 0 );
  }
  
  
  //////////
  // Transmits the given data to the clients.
  inline void sendRenderData( const char * data, 
			      const char * indices,
			      int numDataBytes,
			      int numIndexBytes) {
    char * tmp = scinew char[ numDataBytes + numIndexBytes ];
    std::cerr << "tmp address: " << (void *)tmp << " - " <<
	      (void *)(tmp + numDataBytes + numIndexBytes) << endl;
    // Data first
    //HomogenousConvertHostToNetwork( (void *)tmp,
    //				    (void *)data,
    //				    FLOAT_TYPE,
    //				    numDataBytes );
    NetConversion::convertFloat( (float *)data,
				 numDataBytes / sizeof(float),
				 (float *)tmp );
    // Then indices
    //HomogenousConvertHostToNetwork( (void *)(tmp + numDataBytes),
    //(void *)indices,
    //				    INT_TYPE,
    //				    numIndexBytes );
    NetConversion::convertInt( (int *)indices,
			       numIndexBytes / sizeof(int),
			       (int *)(tmp+numDataBytes) );
    std::cerr << "Converted...";
    //memcpy( tmp, data, numDataBytes );
    //memcpy( tmp + numDataBytes, indices, numIndexBytes );
    sendRenderData( tmp,  numDataBytes + numIndexBytes, true );
    std::cerr << "Sent...";
    //    delete tmp;
    std::cerr << "Deleted!" << endl;
  }
  
  //////////
  // Returns the mailbox for the renderer.
  inline Mailbox<MessageData>& getMailbox() {
    return mailbox;
  }

  ////////
  // Returns the name of this renderer
  virtual const char * const getName() { return name; }

  ////////
  // Returns the version of this renderer
  virtual const char * const getVersion() { return version; }
  
  //////////
  // Name of this renderer.
  static const char * const name;

  //////////
  // Version of this renderer.
  static const char * const version;
  
protected:

  // Protected constructor for sub classes.
  GeometryRenderer( GeometryRenderer * renderer,
		    const char * name,
		    const char * mailboxname ) :
    Renderer( renderer, name ), mailbox( mailboxname, 10 ) {}
  
  // Mailbox for messages from the network.
  Mailbox<MessageData>  mailbox;

  // Number of vertices in the coming data.
  int vertexCount;

  // Number of indices in the coming data.
  int indexCount;

  // Number of polygons in the coming data.
  int polygonCount;

  // 1 if coming data is indexed, 0 if not, -1 if not set.
  int indexed;

  // 1 if coming data is should replace current geom, 0 if not, -1 if not set.
  int replace;

  // Preprocessing function. 
  virtual char * preprocess( const char *data, int &numBytes );
  
  inline void sendViewFrame( const int size, const char * data,
			     const int x, const int y,
			     const int origSize,
			     const int indexed, const int replace,
			     const int vertices=-1, const int indices=-1,
			     const int polygons=-1, const int texture=-1 ) {
    Renderer::sendViewFrame( size, data, x, y, origSize, indexed, replace,
			     vertices, indices, polygons, texture );
  }
  
  // Wrapper to send a 'ViewFrame' message + data to the network.
  // Overrides method in Renderer superclass to allow us to send geom
  // data...
  inline void sendViewFrame( const int size, const char * data,
			     const int x, const int y,
			     const int origSize=-1 ) {
    std::cerr << "Geom sendViewFrame" << endl;
    Renderer::sendViewFrame( size, data, x, y, origSize, indexed, replace,
			     vertexCount, indexCount, polygonCount );
  }

};

}
}

//
// $Log$
// Revision 1.1  2003/07/22 15:46:36  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:34  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.4  2001/08/29 19:55:44  luke
// Fixed ZTexRenderer
//
// Revision 1.3  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.2  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.1  2001/07/16 20:31:02  luke
// Added geometry and ZTex renderers...
//
#endif
