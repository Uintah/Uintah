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

#include <Rendering/ImageRenderer.h>
#include <Network/dataItem.h>
#include <Network/NetConversion.h>
#include <Rendering/RenderGroup.h>

namespace SemotusVisum {
namespace Rendering {

using namespace Network;
using namespace Logging;
using namespace Message;

const char * const
ImageRenderer::name = "Image Streaming";

const char * const
ImageRenderer::version = "$Revision$";

ImageRenderer::ImageRenderer() :
  Renderer( this, "ImageRenderer" ),
  fullX(-1), fullY(-1), offX(-1), offY(-1),
  doSubimage(false),
  mailbox("ImageRendererMailbox", 10 ),
  mouseEventQueueLock("MouseEventQueueLock"),
  mouseEventMutex("MouseEventMutex") {
  
  cerr << "In ImageRenderer::ImageRenderer" << endl;
  mouseEventMutex.lock(); // No current mouse events.
  bps = 3; // Assume RGB data
  cerr << "End of ImageRenderer::ImageRenderer" << endl;
}


ImageRenderer::~ImageRenderer() {

  /* Remove callbacks */
  removeCallbacks();
  
  /* Unlock the mutex */
  mouseEventMutex.unlock();
  
}


void
ImageRenderer::setCallbacks() {
  cerr << "In ImageRenderer::setCallbacks" << endl;
  // Set up our superclass's callbacks.
  setSuperCallbacks();
  
  /* Set up callbacks */
  NetDispatchManager& manager = NetDispatchManager::getInstance();
  
  // Mouse Move
  mouseCallbackID =
    manager.registerCallback( MOUSE_MOVE,
			      ImageRenderer::mouseCallback,
			      this,
			      true );
  cerr << "In ImageRenderer::setCallbacks" << endl;
}

void
ImageRenderer::removeCallbacks() {
  cerr << "In ImageRenderer::removeCallbacks" << endl;
  // Remove the callbacks
  NetDispatchManager& manager = NetDispatchManager::getInstance();

  manager.deleteCallback( mouseCallbackID );
  mouseCallbackID = -1;
  removeSuperCallbacks();
  cerr << "In ImageRenderer::removeCallbacks" << endl;
}

void
ImageRenderer::sendRenderData( const char * data, int numBytes,
			       bool copy ) {
  cerr << "In ImageRenderer::sendRenderData" << endl;
#ifdef PREPROCESS
  if ( copy ) {
    char * _data = scinew char[ numBytes ];
    memcpy( _data, data, numBytes );
    if ( ! helper->getMailbox().trySend( dataItem( _data, numBytes, true ) ) )
      delete[] _data;
  }
  else {
    char * _data = scinew char[ numBytes ];
    memcpy( _data, data, numBytes );
    helper->getMailbox().trySend( dataItem( _data, numBytes, false ) );
    delete[] _data;
  }
#else
  bool needsConvert = true;
  
  if ( group && group->getCompressor() &&
       !group->getCompressor()->needsRGBConvert() )
    needsConvert = false;
    
  if ( copy ) {
    char * _data = scinew char[ numBytes ];
    memcpy( _data, data, numBytes );
    if ( needsConvert )
      NetConversion::convertRGB( _data, numBytes );
    if ( ! helper->getMailbox().trySend( dataItem( _data, numBytes, true ) ) )
      delete[] _data;
  }
  else {
    char * _data = scinew char[ numBytes ];
    memcpy( _data, data, numBytes );
    if ( needsConvert)
      NetConversion::convertRGB( _data, numBytes );
    helper->getMailbox().trySend( dataItem( _data, numBytes, false ) );
    delete[] _data;
  }
#endif

  cerr << "End of ImageRenderer::sendRenderData" << endl;
}

char *
ImageRenderer::resample( const char * image, int width, int height ) {
  cerr << "In ImageRenderer::resample" << endl;
  int scaleSquared = scaling*scaling;
  char * sbuffer = scinew char[ width*height / scaleSquared ];
  for ( int i = 0; i < (width*height*3) / scaleSquared; i +=3 ) {
    sbuffer[ i ] = image[ i * scaling ];
    sbuffer[ i+1 ] = image[ i * scaling + 1 ];
    sbuffer[ i+2 ] = image[ i * scaling + 2 ];
  }

  cerr << "End of ImageRenderer::resample" << endl;

  return sbuffer;
}

void
ImageRenderer::mouseCallback( void * obj, MessageData *input ) {
  cerr << "In ImageRenderer::mouseCallback" << endl;
  MouseMove *mm;
  if ( obj ) {
    mm = (MouseMove *)(input->message);
    if ( mm )
      ( (ImageRenderer *)obj )->mouseInputHandler( *mm );
  }

  cerr << "End of ImageRenderer::mouseCallback" << endl;
}


void
ImageRenderer::mouseInputHandler( MouseMove &mm ) {
  cerr << "In ImageRenderer::mouseInputHandler" << endl;
  /* Get the data from the message */
  int x=0, y=0;
  char button='O';
  int action = MouseMove::UNKNOWN;
  struct timeval eventTime;
  MouseEvent *me = NULL;
  
  gettimeofday( &eventTime, NULL );

  mm.getMove( x, y, button,action );
  
  /* Create a mouseEvent item and place it in the queue */
  me = scinew MouseEvent( x, y, button, action, eventTime );

  mouseEventQueueLock.writeLock();
  mouseEventQueue.push( *me );
  mouseEventQueueLock.writeUnlock();

  /* Unlock the mutex - ie, we have a mouse event */
  mouseEventMutex.unlock();

  cerr << "End of ImageRenderer::mouseInputHandler" << endl;
}

char *
ImageRenderer::preprocess( const char * data, int &numBytes ) {
  cerr << "In ImageRenderer::preprocess" << endl;
#ifdef PREPROCESS
  bool needsConvert = true;
  
  if ( group && group->getCompressor() &&
       !group->getCompressor()->needsRGBConvert() )
    needsConvert = false;
    
  char * _data = scinew char[ numBytes ];
  memcpy( _data, data, numBytes );
  if ( needsConvert )
    NetConversion::convertRGB( _data, numBytes );
  return _data;
#else
  return (char *)data;
#endif
  cerr << "End of ImageRenderer::preprocess" << endl;
}

}
}
