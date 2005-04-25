/*
 *
 * Renderer: Superclass for renderers
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __Renderer_h_
#define __Renderer_h_

#include <stdio.h>
#include <list>

#include <Network/NetInterface.h>
#include <Message/ViewFrame.h>
#include <Thread/Runnable.h>
#include <Logging/Log.h>
#include <Compression/Compression.h>
//#include <Rendering/RenderGroup.h>
#include <Malloc/Allocator.h>

#ifdef __sgi
#pragma set woff 3303
#endif
#define PREPROCESS
namespace SemotusVisum {
namespace Rendering {

using namespace SCIRun;
using namespace Message;
using namespace Network;
using namespace Logging;
using namespace Compression;

class Renderer;

/**************************************
 
CLASS
   RendererHelper
   
KEYWORDS
   Rendering
   
DESCRIPTION

   This is a helper class for Renderer. It runs in its own thread,
   providing the ability to compress data, etc.
      
****************************************/
class RendererHelper : public Runnable {
public:

  //////////
  // Constructor. Initializes our parent.
  RendererHelper( Renderer *parent );

  //////////
  // Destructor.
  ~RendererHelper();

  //////////
  // Run method, which overrides that in the Runnable class.
  virtual void run();

  //////////
  // Returns a reference to our mailbox
#ifndef NO_MBOX
  inline Mailbox<dataItem>& getMailbox() { return mailbox; }
#endif
  
protected:
  // Pointer to our Renderer parent.
  Renderer *parent;

#ifndef NO_MBOX
  // Mailbox
  Mailbox<dataItem> mailbox;
#endif
  
};

class RenderGroup;

/**************************************
 
CLASS
   Renderer
   
KEYWORDS
   Rendering
   
DESCRIPTION

   The rendering superclass defines functions common to all viewing methods.
   
****************************************/
class Renderer {
  friend class RendererHelper;
public:

  //////////
  // Constructor. Does not create an internal rendering thread.
  Renderer() : group( NULL ), internalThread( NULL ), fullMessage( NULL ),
	       fullSize( 0 ), x(0), y(0), bps(1), callbackSet(0)  { }
  
  //////////
  // Constructor. Creates an internal rendering thread.
  Renderer( Renderer * theRenderer, const char * name ) : group( NULL ), 
                                                          fullMessage( NULL ),
                                                          fullSize( 0 ),
                                                          x(0), y(0), bps(1),
                                                          callbackSet(0) {
    helper = scinew RendererHelper( theRenderer );
    internalThread = scinew Thread( helper, name );
  }
  
  //////////
  // Destructor. Clears the list of clients, deletes internal thread.
  virtual ~Renderer() {

    // Clean up our internal buffer.
    if ( fullMessage != NULL )
      delete fullMessage;
  }
  
  ////////
  // Adds callbacks to the network dispatch manager. Should be called
  // before the renderer is used.
  virtual void setCallbacks() = 0;

  ////////
  // Removes callbacks from the network dispatch manager.
  virtual void removeCallbacks() = 0;
  
  //////////
  // Transmits the given data to the clients. If copy is true, the renderer
  // will copy the data. If copy is false, the renderer leaves it. Use the
  // nocopy with care...
  virtual void sendRenderData( const char * data, int numBytes,
			       bool copy=true ) = 0;

  //////////
  // Resets the renderer's parameters.
  virtual void reset() {} 
  
  ////////
  // Callback for compression 
  static void compressCallback( void * obj, MessageData *input );

  ////////
  // Returns the name of this renderer
  virtual const char * const getName() { return name; }

  ////////
  // Returns the version of this renderer
  virtual const char * const getVersion() { return version; }

  //////////
  // Sets the render group for this renderer.
  inline void setRenderGroup( RenderGroup * group ) {
    this->group = group;
  }

  //////////
  // Returns this renderer's render group.
  inline RenderGroup* getRenderGroup() { return group; }
  
  //////////
  // Name of this renderer.
  static const char * const name;

  //////////
  // Version of this renderer.
  static const char * const version;
  
protected:

  // Sets certain callbacks. Should be called by all subclasses...
  void setSuperCallbacks();

  // Removes general renderer callbacks. Should be called by all subclasses.
  void removeSuperCallbacks();
  
  // Wrapper to send a 'ViewFrame' message + data to the network.
  // If origSize is present, it is the uncompressed size of the data.
  virtual void sendViewFrame( const int size, const char * data,
			      const int x, const int y,
			      const int origSize=-1);

  // Wrapper to send a 'ViewFrame' message + data to the network.
  virtual void sendViewFrame( const int size, const char * data,
			      const int x, const int y,
			      const int origSize,
			      const int indexed, const int replace,
			      const int vertices=-1, const int indices=-1,
			      const int polygons=-1, const int texture=-1 );

  virtual void sendViewFrame( const int size, const char * data,
			      const int x, const int y,
			      const int origSize,
			      const int offX, const int offY,
			      const int fullX, const int fullY,
			      const char background[3] );
  
  // Wrapper that transmits data to the network.
  void transmitData( const char * data, const int numBytes );

  // Preprocessing function for subclasses. This function is called in
  // the rendering thread before compression.
  virtual char * preprocess( const char * data, int &numBytes ) = 0;
  
  // Pointer to our rendering group.
  RenderGroup * group;
  
  // Internal thread.
  Thread *internalThread;

  // Internal data buffer
  char * fullMessage;

  // Internal data buffer size
  int    fullSize;

  // Compression callback ID
  int compressionCallbackID;

  // Rendering helper
  RendererHelper *helper;

  // Data span.
  unsigned x,y;

  // Data size (bytes per sample)
  int bps;

private:
  int   callbackSet;
};


}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:38  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:35  simpson
// Adding CollabVis files/dirs
//
// Revision 1.16  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.15  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.14  2001/10/02 01:52:27  luke
// Fixed xerces problem, compression, other issues
//
// Revision 1.13  2001/10/01 18:56:55  luke
// Scaling works to some degree on image renderer
//
// Revision 1.12  2001/08/29 19:55:44  luke
// Fixed ZTexRenderer
//
// Revision 1.11  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.10  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.9  2001/07/16 20:30:02  luke
// Updated renderers...
//
// Revision 1.8  2001/05/29 03:43:13  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.7  2001/05/24 23:03:26  luke
// First transmission successcd ..!
//
// Revision 1.6  2001/05/21 22:00:46  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.5  2001/05/14 18:13:50  luke
// Finished documentation
//
// Revision 1.4  2001/05/14 17:55:36  luke
// New implementation of image renderer - uses a helper class for new thread
//
// Revision 1.3  2001/05/14 17:31:13  luke
// Moved driver to new spot.
//
// Revision 1.2  2001/04/05 21:15:21  luke
// Added header and log info
//
