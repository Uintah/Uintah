/*
 *
 * ImageRenderer: Provides image streaming capability, along with
 *                the ability to deliver mouse events.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __ImageRenderer_h_
#define __ImageRenderer_h_

#include <Rendering/Renderer.h>
#include <Rendering/MouseEvent.h>

#include <Network/NetDispatchManager.h>
#include <Logging/Log.h>

#include <Thread/Mutex.h>
#include <Thread/Mailbox.h>

#include <queue>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {
namespace Rendering {

/**************************************
 
CLASS
   ImageRenderer
   
KEYWORDS
   Rendering, Image Streaming
   
DESCRIPTION

   This class provides an infrastructure for image streaming. It provides
   methods for transmitting images and receiving mouse movements.
   
****************************************/

class ImageRenderer : public Renderer {
public:

  //////////
  // Constructor. Allocates all needed memory.
  ImageRenderer();

  //////////
  // Destructor. Deallocates memory.
  virtual ~ImageRenderer();

  ////////
  // Adds callbacks to the network dispatch manager. Should be called
  // before the renderer is used.
  void setCallbacks();

  ////////
  // Removes callbacks from the network dispatch manager.
  void removeCallbacks();
  
  ////////
  // Callback for mouse events
  static void mouseCallback( void * obj, MessageData *input );

  //////////
  // Transmits the given data to the clients.
  void        sendRenderData( const char * data, int numBytes,
			      bool copy=true );

  //////////
  // Transmits the given data to the clients; the image has the given
  // dimensions.
  inline void sendRenderData( const char * data, int numBytes, unsigned x,
			      unsigned y, bool copy=true ) {
#if 0
    if ( scaling == 1 ) {
      setSize( x, y ); sendRenderData( data, numBytes );
    }
    else {
      int newX = x / scaling;
      int newY = y / scaling;
      setSize( newX , newY );
      int newNumBytes = newX * newY * 3;
      sendRenderData( resample( data, x, y ), newNumBytes );
    }
#else
    setSize( x, y ); sendRenderData( data, numBytes, copy );
#endif
  }

  inline void sendRenderData( const char * data, int numbytes, unsigned x,
			      unsigned y, int offX, int offY, int fullX,
			      int fullY, char bkgd[3],
			      bool copy=true ) {
    this->offX = offX; this->offY = offY; this->fullX = fullX;
    this->fullY = fullY;
    for (int i = 0; i < 3; i++ )
      this->bkgd[i] = bkgd[i];
    sendRenderData( data, numbytes, x, y, copy );
  }
  
  //////////
  // Resets the renderer's parameters.
  virtual void reset() {
    scaling = 1;
  }

  //////////
  // Sets the scaling for this renderer.
  inline void setScale( const int scale ) { scaling = scale; }

  //////////
  // Returns the scaling for this renderer.
  inline int getScale() const { return scaling; }
  
  ///////////
  // Resamples the given image based on the current scale, and returns a new
  // buffer with the resampled image.
  virtual char * resample( const char * image, int width, int height );
  
  //////////
  // Sets the size of the image. 
  inline void setSize( unsigned x, unsigned y ) {
    this->x = x; this->y = y;
  }
  
  //////////
  // Returns the mailbox for the renderer.
  inline Mailbox<MessageData>& getMailbox() {
    return mailbox;
  }

  //////////
  // Returns true if there are mouse events queued up from a client.
  inline bool  mouseEventsQueued() {
    bool result;

    mouseEventQueueLock.readLock();
    result = !mouseEventQueue.empty();
    mouseEventQueueLock.readUnlock();
    return result;
  }

  
  //////////
  // Pops the next mouse event from the queue and returns it.
  inline MouseEvent getMouseEvent() {
    
    mouseEventQueueLock.writeLock();
    
    MouseEvent result = mouseEventQueue.front();
    mouseEventQueue.pop();

    if ( mouseEventQueue.size() == 0 )
      mouseEventMutex.lock();
    
    mouseEventQueueLock.writeUnlock();
    return result;
  }

  //////////
  // Blocks until we have a mouse event.
  inline void waitForMouseEvent() {
    mouseEventMutex.lock();
    mouseEventMutex.unlock();
  }

  inline void setSubimage( const bool subimage ) {
    doSubimage = subimage;
  }

  inline bool getSubimage() const {
    return doSubimage;
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
  int offX, offY, fullX, fullY;
  char bkgd[3];
  bool doSubimage;
  
  inline void sendViewFrame( const int size, const char * data,
			     const int x, const int y,
			     const int origSize=-1 ) {
    Renderer::sendViewFrame( size, data, x, y, origSize, offX, offY,
			     fullX, fullY, bkgd );
  }
  
  // Handles mouse moves.
  void    mouseInputHandler( MouseMove &mm );

  // Preprocessing function. 
  virtual char * preprocess( const char * data, int &numBytes );
  
  // ID for our MouseMove callback.
  int                   mouseCallbackID;

  // Mailbox for messages from the network.
  Mailbox<MessageData>  mailbox;

  // Queue of mouse events.
  queue<MouseEvent>     mouseEventQueue;

  // Lock for mouse event queue.
  CrowdMonitor          mouseEventQueueLock;

  // Mutex for waiting for mouse events.
  Mutex                 mouseEventMutex;

  // Scaling for the renderer.
  int                   scaling;
};

}
}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:36  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:34  simpson
// Adding CollabVis files/dirs
//
// Revision 1.10  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.9  2001/10/01 18:56:54  luke
// Scaling works to some degree on image renderer
//
// Revision 1.8  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.7  2001/07/16 20:30:02  luke
// Updated renderers...
//
// Revision 1.6  2001/05/21 22:00:46  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.5  2001/05/14 18:13:50  luke
// Finished documentation
//
// Revision 1.4  2001/05/14 17:55:35  luke
// New implementation of image renderer - uses a helper class for new thread
//
// Revision 1.3  2001/05/14 17:31:13  luke
// Moved driver to new spot.
//
// Revision 1.2  2001/04/05 21:15:21  luke
// Added header and log info
//
