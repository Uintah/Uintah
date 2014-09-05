/*
 *
 * ZTexRenderer: Provides ZTex creation and transmission capability.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: July 2001
 *
 */


#ifndef __ZTexRenderer_h_
#define __ZTexRenderer_h_

#include <Rendering/GeometryRenderer.h>

#include <Thread/Mailbox.h>

#ifdef PREPROCESS
#include <Thread/CrowdMonitor.h>
#endif

#ifdef __sgi
#pragma set woff 3303
#endif


namespace SemotusVisum {
namespace Rendering {
  

/**************************************
 
CLASS
   ZTexRenderer
   
KEYWORDS
   Rendering, ZTex
   
DESCRIPTION

   This class provides an infrastructure for ZTex creation and transmission.
   It provides methods for transmitting ZTex objects. 
   
****************************************/

class ZTexRenderer : public GeometryRenderer {

public:

  //////////
  // Constructor. Allocates all needed memory.
  ZTexRenderer();

  //////////
  // Destructor. Deallocates memory.
  virtual ~ZTexRenderer();

  ////////
  // Adds callbacks to the network dispatch manager. Should be called
  // before the renderer is used.
  void setCallbacks();

  ////////
  // Removes callbacks from the network dispatch manager.
  void removeCallbacks();
  
  ////////////
  // Sets the OpenGL matrices for reprojecting the vertices.
  void setMatrices( const double * modelview,
		    const double * projection,
		    const int    * viewport );

  ////////////
  // Sets the eyepoint used to render the zbuffer; used to remove any
  // excess trianges from the mesh.
  void setEyepoint( const double * eyePoint );

  ////////////
  // If true, then switches translation in modelview to eyepoint, and sets
  // modelview translation to zero.
  void setEyepointAtZero( bool atZero );
  
  ////////////
  // Sets the dimensions of the z buffer (need not be square!)
  inline void setDimensions( const int width, const int height ) {
    this->width = width; x = width;
    this->height = height; y = height;
  }
  
  //////////
  // Creates a ZTex object and sends it to the clients. numBytes 
  void        sendRenderData( const char * image, const int imageSize,
			      const unsigned int * zbuffer, const int zBufSize,
			      bool copy=true );
  //////////
  // Transmits the given data to the clients. Note - this function
  // probably should not be used. Use the 3 arg version instead.
  void        sendRenderData( const char * data, int numBytes,
			      bool copy=true );
  
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
  double * modelmat;
  double * projmat;
  int    * viewport;
  double * eyept;
  int      width;
  int      height;

#ifdef PREPROCESS
  int imageSize;
  Mutex dataMutex;
#endif
  
  // Preprocessing function. 
  virtual char * preprocess( const char * data, int &numBytes );
  
  inline void sendViewFrame( const int size, const char * data,
			     const int x, const int y,
			     const int origSize,
			     const int indexed, const int replace,
			     const int vertices=-1, const int indices=-1,
			     const int polygons=-1, const int texture=-1 ) {
    GeometryRenderer::sendViewFrame( size, data, x, y, origSize, indexed,
				     replace, vertices, indices, polygons,
				     texture );
  }
  // Wrapper to send a 'ViewFrame' message + data to the network.
  // Overrides method in Renderer superclass to allow us to send geom
  // data...
  inline void sendViewFrame( const int size, const char * data,
			     const int x, const int y,
			     const int origSize ) {
    std::cerr << "ZTex sendViewFrame" << endl;
    Renderer::sendViewFrame( size, data,
			     x, y,
			     origSize, 0, replace,
			     vertexCount, -1, -1, width*height );
  }
};

}
}

//
// $Log$
// Revision 1.1  2003/07/22 15:46:38  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:36  simpson
// Adding CollabVis files/dirs
//
// Revision 1.8  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.7  2001/09/26 17:05:05  luke
// ZTex works.
//
// Revision 1.6  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.5  2001/08/29 19:55:44  luke
// Fixed ZTexRenderer
//
// Revision 1.4  2001/08/24 15:20:02  luke
// Finished ZTexRenderer for driver
//
// Revision 1.3  2001/08/22 16:24:07  luke
// Initial import of ZTex code
//
// Revision 1.2  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.1  2001/07/16 20:31:02  luke
// Added geometry and ZTex renderers...
//
#endif
