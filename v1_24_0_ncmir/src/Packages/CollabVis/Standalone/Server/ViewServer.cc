/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  ViewServer.h: Serves data to remote clients
 *
 *  Written by:
 *   Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   August 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Malloc/Allocator.h> // for scinew
#include <Packages/CollabVis/Standalone/Server/ViewServer.h>
#include <Packages/CollabVis/Standalone/Server/Network/NetInterface.h>
#include <Packages/CollabVis/Standalone/Server/Network/NetDispatchManager.h>
#include <Packages/CollabVis/Standalone/Server/Message/MouseMove.h>
#include <Packages/CollabVis/Standalone/Server/Logging/Log.h>
#include <Packages/CollabVis/Standalone/Server/Rendering/Renderer.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
//#include <Dataflow/Modules/Render/Renderer.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Modules/Render/OpenGL.h>
#include <Packages/CollabVis/Standalone/Server/Rendering/ZTex/Image.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Connection.h>
// tmp
#include <Packages/CollabVis/Standalone/Server/Compression/LZO.h>

// These damn things weren't defined in a header file... :(
const int MouseStart = 0;
const int MouseEnd = 1;
const int MouseDrag = 2;

namespace SCIRun {
#define PARALLEL_RENDERING

using namespace SemotusVisum::Network;
using namespace SemotusVisum::Rendering;
using namespace SemotusVisum::Message;
using namespace SemotusVisum::Logging;

//const char * parallelDisplays[] = {":3.0", ":4.0", ":5.0"};
//const int numParallelDisplays = 3;
const char * parallelDisplays[] = {":4.0"};
const int numParallelDisplays = 1;

ViewServerThread::ViewServerThread( ViewServerHelper *parent ) :
  parent( parent ), mailbox( "ViewServerThreadMailbox", 5 ),
  runLock( "ViewServerThreadRunLock" )
{

}

ViewServerThread::~ViewServerThread() {

}

void
ViewServerThread::run() {
  RequestAndWindow raw;
  
  for (;;) {
    raw = mailbox.receive();
    if ( raw.info == NULL )
      return; // Our parent wants us to go away...

    // Handle the data request
    parent->handleDataRequest( raw.info, raw.request );

    // Unlock us, since we're guaranteed to be locked when we service a
    // request...
    runLock.unlock();
  }
}

// Note - this function is not 100% thread-safe. It checks availability by
// looking at the runlock, then unlocking it if it can lock it. It is
// possible that someone could grab the lock in between calling this function
// and sending a request. Caveat emptor...
bool
ViewServerThread::isAvailable() {
  bool returnval = runLock.tryLock();
  if ( returnval ) runLock.unlock();
  return returnval;
}

bool
ViewServerThread::sendRequest( RenderRequest request, ViewWindowInfo *info ) {
  char buffer[ 1000 ];
  
  if ( !runLock.tryLock() ) {   
    snprintf( buffer, 1000, "Unable to get lock for server thread 0x%x!",
	      (unsigned)this );
    Log::log( WARNING, buffer );
    return false;
  }

  RequestAndWindow raw;
  raw.request = request;
  raw.info = info;
  if ( !mailbox.trySend( raw ) ) {
    snprintf( buffer, 1000, "Could not send request to server thread 0x%x!",
	      (unsigned)this );
    Log::log( WARNING, buffer );
    runLock.unlock();
    return false;
  }
  
  return true;
}

ViewServerHelper::ViewServerHelper() : renderQueue( "RenderQueue", 100 ),
				       renderWindowMapLock( "RWMLock" ),
				       parallelWindowLock( "ParallelWinLock" )
#ifdef RENDER_QUEUE
  , requestQueueLock("RequestQueueLock")
#endif
{
  
}

ViewServerHelper::~ViewServerHelper() {

}

void
ViewServerHelper::sendRenderRequest( RenderRequest request ) {
  Log::log( SemotusVisum::Logging::DEBUG, "Adding render request..." );
  renderQueue.send( request );
}

void
ViewServerHelper::run() {

  RenderRequest renderRequest;
  ViewWindowInfo * window = NULL;
  char buffer[ 1000 ];
  WallClockTimer timer;
  
#ifdef RENDER_QUEUE
  for ( ;; ) {
    
    // See if our queues are empty.
    bool allQueuesEmpty = true;
    requestQueueLock.lock();
    for ( int i = 0; i < requestQueues.size(); i++ ) 
      if ( !requestQueues[i].empty() ) {
	allQueuesEmpty = false;
	break;
      }
    requestQueueLock.unlock();

    // If all queues are empty, we wait for the next request.
    bool gotRequest;
    if ( allQueuesEmpty ) {
      renderRequest = renderQueue.receive();
      gotRequest = true;
      Log::log( SemotusVisum::Logging::DEBUG, "All queues empty!" );
      
    }
    else {
      Log::log( SemotusVisum::Logging::DEBUG, "Not all queues empty!" );
	  
      // Otherwise, we just check our mailbox.
      gotRequest = renderQueue.tryReceive( renderRequest );
    }
    timer.clear();
    timer.start();
    // If we got a request, then add it to the appropriate queue.
    if ( gotRequest ) {

      Log::log( SemotusVisum::Logging::DEBUG, "Got a request!" );

      // If we get a request with null info, it's a 'die' request.
      if ( renderRequest.groupInfo == NULL ) {
        Log::log( SemotusVisum::Logging::MESSAGE, "Helper dying!" );
	RenderRequest r;
	for ( int i = 0; i < helperThreads.size(); i++ )
	  while ( !helperThreads[i]->sendRequest( r, NULL ) );
	helperThreads.remove_all();
	return;
      }
      
      // Find the appropriate view window
      window = getWindow( renderRequest.groupInfo );
      
      // Add it to the queue
      if ( window != NULL ) {
	Log::log( SemotusVisum::Logging::DEBUG, "Got a window!" );
	requestQueueLock.lock();
	for ( int i = 0; i < requestQueueIDs.size(); i++ )
	  if ( requestQueueIDs[i] == window ) {
	    snprintf( buffer, 1000, "Adding request 0x%x to queue 0x%x",
	          (unsigned)renderRequest.groupInfo, (unsigned)window );
	    Log::log( SemotusVisum::Logging::DEBUG, buffer );
	    requestQueues[ i ].push( renderRequest );
	    break;
	  }
	requestQueueLock.unlock();
      }
    }

    // Now, we run through the queues, trying to dispatch a request for each
    // queue.
    requestQueueLock.lock();
    for ( int i = 0; i < requestQueues.size(); i++ ) {
      renderRequest = requestQueues[i].front();
      
      // Allocate a window for rendering.
      window = allocateWindow( renderRequest.groupInfo );

      // If we get a window, dispatch a renderer.
      if ( window != NULL ) {
	requestQueues[i].pop();
	dispatchHelper( window, renderRequest );
      }
      // Otherwise, do nothing.
	
    }
      
    requestQueueLock.unlock();
    timer.stop();
    snprintf( buffer, 1000,
	      "Took %0.2f seconds to process request", timer.time() );
    Log::log( SemotusVisum::Logging::MESSAGE, buffer );
  }
#else
  for (;;) {
    Log::log( SemotusVisum::Logging::DEBUG, "Waiting for render request..." );
    renderRequest = renderQueue.receive();
    timer.clear();
    timer.start();
    Log::log( SemotusVisum::Logging::DEBUG, "Got a render request!" );
#ifndef PARALLEL_RENDERING
    if ( renderRequest.groupInfo == NULL )
      return; // The server wants us dead!

    // Find the appropriate data view window
    /* This is the old, nonparallel way of doing it!
       window = getWindow( renderRequest.groupInfo ); */
    window = allocateWindow( renderRequest.groupInfo );
    
    if ( window == NULL )
      continue; // Punt, if we don't have a data window.

    // Handle the data request
    handleDataRequest( window, renderRequest );
#else
    // If we get a request with null info, it's a 'die' request.
    if ( renderRequest.groupInfo == NULL ) {
      RenderRequest r;
      for ( int i = 0; i < helperThreads.size(); i++ )
	while ( !helperThreads[i]->sendRequest( r, NULL ) );
      helperThreads.remove_all();
      return;
    }

    // Allocate a window for rendering.
    window = allocateWindow( renderRequest.groupInfo );

    // Now, depending on the request type, we'll either punt or requeue.
    // If it's an image request, just drop the frame (cuz we don't want to
    // start getting frames out of order). Order doesn't matter on the others,
    // so requeue the request.
    if ( window == NULL )
      if ( renderRequest.renderType == IMAGE ) {
	Log::log( SemotusVisum::Logging::DEBUG,
		  "Dropping image request!" );
	continue;
      }
      else {
	// only try - don't want to block the only thread that reads from
	// this mailbox!
	/*Log::log( SemotusVisum::Logging::DEBUG,
		  "Trying to resend non-image request!" );
	if ( !renderQueue.trySend( renderRequest ) )
	  Log::log( SemotusVisum::Logging::WARNING,
	  "Render queue full! Dropping request" );*/
	continue;
      }

    // Grab a thread and send a request to it!
    dispatchHelper( window, renderRequest );
    timer.stop();
    snprintf( buffer, 1000,
	      "Took %0.2f seconds to process request", timer.time() );
    Log::log( SemotusVisum::Logging::MESSAGE, buffer );
#endif
  }
#endif // Render queue
}

ViewWindowInfo *
ViewServerHelper::getDataWindow( RenderGroupInfo * groupInfo ) {
  Log::log(  SemotusVisum::Logging::DEBUG, "(ViewServerHelper::getDataWindow) entered\n" );
  ViewWindowInfo * returnval = NULL;
  map<RenderGroupInfo *,ViewWindowInfo *>::iterator location;
  
  renderWindowMapLock.lock();

  // This is a hack for map.find()...
  for ( location = renderWindowMap.begin() ;
	location != renderWindowMap.end() ;
	location++ )
    if ( location->first == groupInfo && location->second->allocate() ) {
      returnval = location->second;
      break;
    }
  renderWindowMapLock.unlock();
  //cerr << "Renderer " << groupInfo << " maps to window ";
  //if ( returnval == NULL ) cerr << "NONE!" << endl;
  //else cerr << returnval->window << endl;
  Log::log(  SemotusVisum::Logging::DEBUG, "(ViewServerHelper::getDataWindow) leaving\n" );
  return returnval;
}

ViewWindowInfo *
ViewServerHelper::getParallelWindow( RenderGroupInfo * groupInfo ) {
  ViewWindowInfo * info = NULL;
  ViewWindowInfo * returnVal = NULL;
  char buffer[1000];
  
  info = getWindow( groupInfo );
  
  if ( info == NULL ) {
    snprintf( buffer, 1000, "Cannot get data window for render group 0x%x",
	      (unsigned)groupInfo );
    Log::log( SemotusVisum::Logging::ERROR, buffer );
    return NULL;
  }
  
  parallelWindowLock.readLock();
  for ( int i = 0; i < parallelWindows.size(); i++ ) {
    // If they belong to the same viewer, see if we can use it.
    if ( info->window->manager == parallelWindows[i]->window->manager )
      if ( parallelWindows[i]->allocate() ) {
	returnVal = parallelWindows[i];
	break;
      }
  }
  parallelWindowLock.readUnlock();

  
  if ( returnVal == NULL )
    snprintf( buffer, 1000, "Cannot get parallel window for group 0x%x",
	      (unsigned)groupInfo );
  else 
    snprintf( buffer, 1000, "Got parallel window 0x%x for group 0x%x",
	      (unsigned)returnVal, (unsigned)groupInfo );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  return returnVal;
}

ViewWindow *
ViewServerHelper::createNewViewWindow( ViewWindow *window, int whichDisplay ){
  // Create the appropriate command
  string viewerName = window->manager->getID();
  //cerr << "CNVW: Viewer name == " << viewerName << endl;
  string display = parallelDisplays[ whichDisplay ];
  //cerr << "CNVW: Display == " << display << endl;
  string command = viewerName + " initialize_ui " + display;
  //cerr << "CNVW: Command == " << command << endl;
  
  // Execute the tcl command
  string str_result;
  Log::log( SemotusVisum::Logging::DEBUG, "Calling TCL::eval()");
  int result = window->manager->getGui()->eval( command, str_result );
  Log::log( SemotusVisum::Logging::DEBUG, "Done calling TCL::eval()");
  
  if ( !result ) {
    Log::log( SemotusVisum::Logging::DEBUG,
	      "Failure creating new view window!" );
    Log::log( SemotusVisum::Logging::DEBUG, ccast_unsafe(str_result) );
    return NULL;
  }
  else {
    // return the new view window.
    Log::log( SemotusVisum::Logging::DEBUG,
	      "Success creating new view window!" );
    if ( str_result != "" )
      Log::log( SemotusVisum::Logging::DEBUG, ccast_unsafe(str_result) );
    
    // Get our view window from the Viewer
    ViewWindow * newWindow = window->manager->newViewWindowMailbox.receive();

    char buffer[100];
    snprintf( buffer, 100, "New View Window: 0x%x", (unsigned)newWindow );
    Log::log( SemotusVisum::Logging::MESSAGE, buffer );
    return newWindow;
  }
}

ViewWindowInfo *
ViewServerHelper::createParallelWindow( RenderGroupInfo * groupInfo ) {
  char buffer[1000];

  // FOR RIGHT NOW!!!!
  //return NULL;
  
  // If we don't have any more windows available, punt.
  Log::log( SemotusVisum::Logging::DEBUG, "Locking parallel win list\n" );
  parallelWindowLock.writeLock();
  Log::log( SemotusVisum::Logging::DEBUG, "Locked parallel win list\n" );
  if ( parallelWindows.size() >= numParallelDisplays ) {
    parallelWindowLock.writeUnlock();
    snprintf( buffer, 1000,
          "Cannot create another parallel window for group 0x%x",
          (unsigned)groupInfo );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    return NULL;
  }

  // Allocate a parallel window with the same viewer as our data window.
  Log::log( SemotusVisum::Logging::DEBUG,
	    "Getting normal data window for parallel win\n" );
  ViewWindowInfo *dataWindow = getWindow( groupInfo );
  Log::log( SemotusVisum::Logging::DEBUG, "Got NDW" );
  snprintf( buffer, 1000, "Parallel%s", dataWindow->name );
  ViewWindow * window = createNewViewWindow( dataWindow->window,
					     parallelWindows.size() );
  Log::log( SemotusVisum::Logging::DEBUG, "Done createNewViewWindow" );
  if ( window == NULL ) {
    parallelWindowLock.writeUnlock();
    snprintf( buffer, 1000, "Unable to create new view window for group 0x%x",
          (unsigned)groupInfo );
    Log::log( SemotusVisum::Logging::WARNING, buffer );
    return NULL;
  }
  ViewWindowInfo * info = scinew ViewWindowInfo( window, buffer );

  parallelWindows.add( info );
  snprintf( buffer, 1000, "Created parallel window 0x%x for group 0x%x",
      (unsigned)info, (unsigned)groupInfo );
  Log::log( SemotusVisum::Logging::MESSAGE, buffer );
  parallelWindowLock.writeUnlock();
  return info;
}


ViewWindowInfo *
ViewServerHelper::allocateWindow( RenderGroupInfo * groupInfo ) {
  ViewWindowInfo * window = NULL;
  char buffer[1000];
  
  // If the regular data window is available, allocate and return that.
  Log::log( SemotusVisum::Logging::DEBUG, "Trying data window" );
  if ( (window = getDataWindow( groupInfo )) ) {
    snprintf( buffer, 1000, "Got data window 0x%x for group 0x%x\n",
	      (unsigned)window, (unsigned)groupInfo );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    return window;
  }
#ifdef FULL_PARALLEL_RENDERING 
  // Else, if we have a parallel data window available for that viewer,
  // allocate and use that.
  Log::log( SemotusVisum::Logging::DEBUG, "Trying parallel window" );
  if ( window = getParallelWindow( groupInfo ) ) {
    snprintf( buffer, 1000, "Got parallel window 0x%x for group 0x%x\n",
	      window, groupInfo );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    return window;
  }
  
  // Else, if we have parallel data windows available, create one. Allocate
  // and return that.
  Log::log( SemotusVisum::Logging::DEBUG, "Creating parallel window" );
  if ( window = createParallelWindow( groupInfo ) ) {
    snprintf( buffer, 1000, "Created parallel window 0x%x for group 0x%x\n",
	      window, groupInfo );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    return window;
  }
#endif
  // Else, punt.
  else {
    snprintf( buffer, 1000, "Unable to allocate window for group 0x%x\n",
	      (unsigned)groupInfo );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    return NULL;
  }
}


void
ViewServerHelper::addRenderWindowMap( RenderGroupInfo * groupInfo,
				      ViewWindowInfo * info ) {
  //cerr << "Adding render window map: " << groupInfo << " to window " <<
  //  info->window << endl;
  
  renderWindowMapLock.lock();
  
  // See if this is a new window. If so, create a thread for it.
  map<RenderGroupInfo *,ViewWindowInfo *>::iterator location;
  bool newWindow = true;
  for ( location = renderWindowMap.begin() ;
	location != renderWindowMap.end() ;
	location++ )
    if ( location->second == info ) {
      newWindow = false;
      break;
    }
  if ( newWindow ) {
    // Create thread
    //cerr << "Creating thread for window " << info->window << endl;
    ViewServerThread *vst = scinew ViewServerThread( this );
    Thread * t = scinew Thread( vst, "ViewServerThread" );
    t->detach();
    helperThreads.add( vst );

#ifdef RENDER_QUEUE
    // Create a new queue
    char buffer[1000];
    snprintf( buffer, 1000, "Adding new queue with ID 0x%x", (unsigned)info );
    Log::log( SemotusVisum::Logging::DEBUG, buffer );
    
    requestQueueLock.lock();
    queue<RenderRequest> q;
    requestQueues.add( q );
    requestQueueIDs.add( info );
    requestQueueLock.unlock();
#endif
  }
  renderWindowMap.insert( make_pair( groupInfo, info ) );
  renderWindowMapLock.unlock();

  char * group = groupInfo->group->getName();
  char * viewer = info->name;
  
  GroupViewer gv(false);
  gv.groupAdded( group, viewer );
  gv.finish();
  NetInterface::getInstance().sendPriorityDataToAllClients( gv );

}

void
ViewServerHelper::removeRenderWindowMap( RenderGroupInfo * groupInfo,
					 ViewWindowInfo * info ) {
  
  map<RenderGroupInfo *,ViewWindowInfo *>::iterator location;
  
  renderWindowMapLock.lock();
  location = renderWindowMap.find( groupInfo );
  if ( location != renderWindowMap.end() ) {
    //cerr << "Removing render window map: " << groupInfo << " to window " <<
    //  info->window << endl;
    renderWindowMap.erase( location );
  }

  // If there are no more groups mapped to this window, kill a thread.
  bool noMoreGroups = true;
  for ( location = renderWindowMap.begin() ;
	location != renderWindowMap.end() ;
	location++ )
    if ( location->second == info ) {
      noMoreGroups = false;
      break;
    }
  if ( noMoreGroups ) {
    //cerr << "Deleting thread for window " << info->window << endl;
    // FIXME - memory leak, not deleting thread.
    RenderRequest r;
    while ( !helperThreads[helperThreads.size()-1]->sendRequest( r, NULL ) );
    helperThreads.remove( helperThreads.size()-1 );

#ifdef RENDER_QUEUE
    // Also remove the appropriate render queue
    bool found = false;
    char buffer[ 1000 ];
    requestQueueLock.lock();
    for ( int i = 0; i < requestQueueIDs.size(); i++ ) 
      if ( requestQueueIDs[ i ] == info ) {
	requestQueueIDs.remove( i );
	requestQueues.remove( i );
	found = true;
	break;
      }
    if ( !found ) {
      snprintf( buffer, 1000, "Cannot remote request queue with ID 0x%x",
		(unsigned)info );
      Log::log( SemotusVisum::Logging::ERROR, buffer );
    }
    requestQueueLock.unlock();
#endif
  }
  renderWindowMapLock.unlock();

  char * group = groupInfo->group->getName();
  char * viewer = info->name;

  GroupViewer gv(false);
  gv.groupSubtracted( group, viewer );
  gv.finish();
  NetInterface::getInstance().sendPriorityDataToAllClients( gv );
}

ViewWindowInfo *
ViewServerHelper::getWindow( const RenderGroupInfo * groupInfo ) {
  ViewWindowInfo * returnval = NULL;
  map<RenderGroupInfo *,ViewWindowInfo *>::iterator location;
  
  renderWindowMapLock.lock();

  // This is a hack for map.find()...
  for ( location = renderWindowMap.begin() ;
	location != renderWindowMap.end() ;
	location++ )
    if ( location->first == groupInfo ) {
      returnval = location->second;
      break;
    }
  renderWindowMapLock.unlock();
  // ANNOYING ! cerr << "Renderer " << groupInfo << " maps to window ";
  //if ( returnval == NULL ) cerr << "NONE!" << endl;
  //else cerr << returnval->window << endl;
  return returnval;
}

void
ViewServerHelper::handleDataRequest( ViewWindowInfo * info,
				     RenderRequest request ) {
  
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewServerHelper::handleDataRequest) entered \n" );
  char buffer[1000];
  ViewWindow * window = info->window;
  WallClockTimer timer;
  timer.start();
  switch ( request.renderType ) {
  case IMAGE:
    {
      snprintf( buffer, 1000,
		"Doing an image request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) locking groupInfoLock 1"  );
      
      //window->groupInfoLock.lock();
      if( !window->groupInfoLock.tryLock() )
      {
        Log::log( SemotusVisum::Logging::WARNING, "(ViewServerHelper::handleDataRequest) failed to lock groupInfoLock 1"  );
	info->release();
	return;
      }
      
      window->groupInfo = request.groupInfo;

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) unlocking groupInfoLock 1"  );
      
      window->groupInfoLock.unlock();
      
      window->force_redraw();
      window->redraw_if_needed();

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) locking groupInfoLock 2"  );
      
      //window->groupInfoLock.lock();
      if( !window->groupInfoLock.tryLock() )
      {
        Log::log( SemotusVisum::Logging::WARNING, "(ViewServerHelper::handleDataRequest) failed to lock groupInfoLock 2"  );
	info->release();
	return;
      }
      window->groupInfo = NULL;

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) unlocking groupInfoLock 2"  );
      
      window->groupInfoLock.unlock();
      snprintf( buffer, 1000,
		"Done with image request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      timer.stop();
      snprintf( buffer, 1000,
		"Image rendering time was %0.2f seconds",
		timer.time() );
      Log::log( SemotusVisum::Logging::MESSAGE, buffer );
    }
    break;
  case GEOM:
    {
      snprintf( buffer, 1000,
		"Doing a geom request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      GeometryRenderer *geomRenderer =
	(GeometryRenderer *)request.groupInfo->group->getRenderer();
      FutureValue<GeometryData*> result("GeometryData");
      int datamask = GEOM_TRIANGLES|GEOM_VIEW;
      window->getData( datamask, &result );
      GeometryData *gd = result.receive();

      // Set the view
      Log::log( SemotusVisum::Logging::DEBUG, "Setting view for geom request");
      SetViewingMethod svm( false ); // Reply    
      svm.setOkay( true, GeometryRenderer::name ); // Okay to switch
      View *view = gd->view;
      Point eye = view->eyep();
      Point at = view->lookat();
      Vector up = view->up();
      
      
      // This really ought to be a geom...
      VMImageStreaming * vmi = scinew VMImageStreaming();
      
      vmi->setEyePoint( eye.x(), eye.y(), eye.z() );
      vmi->setLookAtPoint( at.x(), at.y(), at.z() );
      vmi->setUpVector( up.x(), up.y(), up.z() );
      vmi->setPerspective( view->fov(), gd->znear, gd->zfar );
      svm.setImageStream( vmi );
      svm.finish(); // Finish generating message
      // Send the message
      NetInterface::getInstance().
	sendPriorityDataToClients( geomRenderer->getRenderGroup()->
				   getClients(),
				   svm );
    
      /* Get and transmit the geometry */
      Array1<float> * triangles = (Array1<float> *)(gd->depthbuffer);
      std::cerr << "# Triangles: " << (triangles->size()/3) << endl;
      float * data = triangles->get_objs();

      /* TEMP - See how much this compresses! */
#ifdef DUMB_SHIT
      LZOCompress lzo;
      int lzosize;
      DATA * out = scinew DATA[ triangles->size() * sizeof(float) ];
      lzosize = lzo.compress( (DATA*)data, 1,
			      triangles->size() * sizeof(float),
			      &out );
      std::cerr << "Compressed size is " << lzosize << endl;
      std::cerr << "Uncompressed size was " <<
	triangles->size() * sizeof(float) << endl;
      delete[] out;
#endif
      /* End of temp */

      // TMP - WRITE TRIS TO DISK
      {
	FILE * f = fopen( "tris.out", "w" );
	for ( int i = 0; i < triangles->size(); i+= 3 )
	  fprintf( f, "Triangle %d: (%0.2f, %0.2f, %0.2f)\n",
		   i / 3 + 1, data[i], data[i+1], data[i+2] );
	fclose( f );
      }
      // /TMP
      geomRenderer->setVertexCount( triangles->size()/3 );
      geomRenderer->setIndexed( false );
      geomRenderer->setReplace( true ); // For right now.
      // Convert the data. We know it's float data...
      float * fnew = scinew float[ triangles->size() ];
      //HomogenousConvertHostToNetwork( (void *)data,
      //				      (void *)data,
      //				      FLOAT_TYPE,
      //				      triangles->size()/3 );
      
      //geomRenderer->GeometryRenderer::sendRenderData( (char *)data,
      //						      triangles->size() * sizeof(float) );
      snprintf( buffer, 1000, "Converting %d bytes",
		triangles->size() );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      HomogenousConvertHostToNetwork( (void *)fnew,
				      (void *)data,
				      FLOAT_TYPE,
				      triangles->size() );
      
      geomRenderer->GeometryRenderer::sendRenderData( (char *)fnew,
						      triangles->size() *
						      sizeof(float) );
      delete fnew; // try ej
      delete triangles;     
      snprintf( buffer, 1000,
		"Done with geom request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      timer.stop();
      snprintf( buffer, 1000,
		"Geometry rendering time was %0.2f seconds",
		timer.time() );
      Log::log( SemotusVisum::Logging::MESSAGE, buffer );
    }
    break;
  case ZTEX:
    {
      snprintf( buffer, 1000,
		"Doing a ztex request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      getZTex( request, info );
      snprintf( buffer, 1000,
		"Done with ztex request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      timer.stop();
      snprintf( buffer, 1000,
		"ZTex rendering time was %0.2f seconds",
		timer.time() );
      Log::log( SemotusVisum::Logging::MESSAGE, buffer );
    }
    break;
  case VIEW: // This isn't a rendering request; rather, just a request for
             // a view...
    {
      char * client = ( request.clientName == NULL ?
			strdup("Null") :
			strdup(request.clientName) );
      snprintf( buffer, 1000,
		"Doing a view request for client %s, on window 0x%x",
		client, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      
      FutureValue<GeometryData*> result("ViewData");
      int datamask = GEOM_VIEW;
      window->getData( datamask, &result );
      GeometryData *gd = result.receive();
      SetViewingMethod setvm( false ); // Reply
      setvm.setOkay( true,
		     request.groupInfo->group->getRenderer()->getName() );
      View *view = gd->view;
      Point eye = view->eyep();
      Point at = view->lookat();
      Vector up = view->up();
      
      // This really ought to be a geom...
      VMImageStreaming * vmi = scinew VMImageStreaming();
      
      vmi->setEyePoint( eye.x(), eye.y(), eye.z() );
      vmi->setLookAtPoint( at.x(), at.y(), at.z() );
      vmi->setUpVector( up.x(), up.y(), up.z() );
      vmi->setPerspective( view->fov(), gd->znear, gd->zfar );
      setvm.setImageStream( vmi );
      setvm.finish(); // Finish generating message 

      // Send the message
      if ( request.clientName != NULL ) {
	Log::log( SemotusVisum::Logging::DEBUG,
		  "Sending view info to single client" );
	NetInterface::getInstance().
	  sendPriorityDataToClient( request.clientName,
				    setvm );
      }
      else {
	Log::log( SemotusVisum::Logging::DEBUG,
		  "Sending view info to render group" );
	NetInterface::getInstance().
	  sendPriorityDataToClients( request.groupInfo->group->getClients(),
				     setvm );
      }
      snprintf( buffer, 1000,
		"Done with view request for client %s on window 0x%x",
		client, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      timer.stop();
      snprintf( buffer, 1000,
		"View rendering time was %0.2f seconds",
		timer.time() );
      Log::log( SemotusVisum::Logging::MESSAGE, buffer );
    }
  case MOUSE: // We have mouse move info.
    {
      snprintf( buffer, 1000,
		"Doing a mouse request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      // We get our mouse events from an opposite coordinate frame in y.
      // Thus, we have to flip them.
      // Not true anymore! ejl 10-25
      //int y = window->current_renderer->xres - request.me.y;
      int y = request.me.y;

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) locking groupInfoLock 3"  );
      
      //string foo("foo");
      
      //window->groupInfoLock.lock();
      if( !window->groupInfoLock.tryLock() )
      {
        Log::log( SemotusVisum::Logging::WARNING, "(ViewServerHelper::handleDataRequest) failed to lock groupInfoLock 3"  );
	info->release();
	return;
      }
      
      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before request.groupInfo"  );
       
      window->groupInfo = request.groupInfo;

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before setViewState" );
      
      //cerr << "Before: " << endl;
      //window->groupInfo->state.view.emit( cerr, foo );

      window->setViewState( window->groupInfo->state );

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) unlocking groupInfoLock 3"  );
       
      window->groupInfoLock.unlock();

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before request.me.button L" );
      
      // Request.me.button - I'll request *your* button, baby...
      if ( request.me.button == 'L' )
      {
	Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before window->mouse_translate" );
	window->mouse_translate( request.me.action,
				 request.me.x,
				 y, 0, 0, 0 );
      }
      else if ( request.me.button == 'M' )
      {
	Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before window->mouse_rotate" );
	window->mouse_rotate( request.me.action,
			      request.me.x,
			      y, 0, 0, request.me.timeMS );
      }
      else if ( request.me.button == 'R' )
      {
	Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before window->mouse_scale" );
	window->mouse_scale( request.me.action,
			     request.me.x, y,0,0,0 );
      }
      else 
	Log::log( SemotusVisum::Logging::ERROR, "Unknown mouse button!" );

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) locking groupInfoLock 4"  );
      
      //window->groupInfoLock.lock();
      if( !window->groupInfoLock.tryLock() )
      {
        Log::log( SemotusVisum::Logging::WARNING, "(ViewServerHelper::handleDataRequest) failed to lock groupInfoLock 4"  );
	info->release();
	return;
      }

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) before getViewState"  );
      
      window->getViewState( window->groupInfo->state );

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) unlocking groupInfoLock 4"  );
      
      window->groupInfoLock.unlock();
      
      window->redraw_if_needed();
      //cerr << "After: " << endl;
      //window->groupInfo->state.view.emit( cerr, foo );
      //cerr << "==================================" << endl;

      // Special - if the window is in inertia mode, add another render
      // request.
      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) locking groupInfoLock 5"  );
      
      //window->groupInfoLock.lock();
      if( !window->groupInfoLock.tryLock() )
      {
        Log::log( SemotusVisum::Logging::WARNING, "(ViewServerHelper::handleDataRequest) failed to lock groupInfoLock 5"  );
	info->release();
	return;
      }
      if ( window->groupInfo->state.inertia_mode )
	sendRenderRequest( request );
      
      window->groupInfo = NULL;

      Log::log( SemotusVisum::Logging::MESSAGE, "(ViewServerHelper::handleDataRequest) unlocking groupInfoLock 5"  );
      
      window->groupInfoLock.unlock();
  
      snprintf( buffer, 1000,
		"Done with mouse request for renderer 0x%x on window 0x%x",
		(unsigned)request.groupInfo, (unsigned)window );
      Log::log( SemotusVisum::Logging::DEBUG, buffer );
      timer.stop();
      snprintf( buffer, 1000,
		"Mouse rendering time was %0.2f seconds",
		timer.time() );
      Log::log( SemotusVisum::Logging::MESSAGE, buffer );
      break;
    }
  default:
    {
      snprintf(buffer,1000,"Unknown render request %d", request.renderType );
      Log::log( SemotusVisum::Logging::ERROR, buffer );
    }
  }
  // Release our allocated window.
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewServerHelper::handleDataRequest) releasing allocated window\n" ); 
  info->release();

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewServerHelper::handleDataRequest) leaving \n" );
}

void
ViewServerHelper::dispatchHelper( ViewWindowInfo * window,
				  RenderRequest request ) {
  // Try to get a thread. We'll always have the same number of threads as
  // we do windows, so if we don't get one within 5 passes we've found a bug.
  // Thus, we'll report this in the log, and drop the request.

  for ( int numTries = 0; numTries < 5; numTries++ )
    for ( int i = 0; i < helperThreads.size(); i++ )
      if ( helperThreads[i]->sendRequest( request, window ) )
	return; // Success!
  
  char buffer[1000];
  snprintf( buffer, 1000,
	    "Unable to dispatch helper for request 0x%x, window 0x%x!",
	    (unsigned)&request, (unsigned)window );
  Log::log( SemotusVisum::Logging::ERROR, buffer );
}


void
ViewServerHelper::getZTex( const RenderRequest &request,
			   ViewWindowInfo * info ) {
  double eye[3] = {0,0,-2};
  double at[3] = {0,0,2};
  double up[3] = {0,1,0};
  double * matrix = NULL;
  bool matrixMode = false;
  ViewWindow * window = info->window;
  GetZTex * gzt = (GetZTex * )(request.gzt);

  /* Get the data from the message */
  gzt->getEyePoint( eye[0], eye[1], eye[2] );

  if ( gzt->isMatrixSet() ) {
    matrix = gzt->getMatrix();
    matrixMode = true;
  }
  else {
    gzt->getLookAtPoint( at[0], at[1], at[2] ); 
    gzt->getUpVector( up[0], up[1], up[2] );
  }

  cerr << "Eyepoint is " << eye[0] << " " << eye[1] << " " << eye[2] << endl;
  if ( matrixMode )
    for (int i = 0; i < 16; i++ ) {
      if ( (i+1) % 4 == 0 ) cerr << endl;
      cerr << matrix[i] << " ";
    }
  else {
    cerr << "Look at: " << at[0] << " " << at[1] << " " << at[2] << endl;
    cerr << "Up: " << up[0] << " " << up[1] << " " << up[2] << endl; 
  }

  // Redraw the scene, so that everything is current
  if ( matrixMode )
    ((OpenGL *)(window->current_renderer))->setZTexTransform( matrix );
  else {
    Point p1(eye[0], eye[1], eye[2]);
    Point p2(at[0], at[1], at[2]);
    Vector v1(up[0], up[1], up[2]);
    View v(p1, p2, v1, 0.666 );
	     //View v( Point( eye[0], eye[1], eye[2] ),
	     //Point( at[0], at[1], at[2] ),
	     //Vector( up[0], up[1], up[2] ),
	     //0.666 /* Doesn't matter! */);
	     
    ((OpenGL *)(window->current_renderer))->setZTexView( v );
  }
  Log::log( MESSAGE, "*1* Set ztex view!" );
  
  // TURN OFF LIGHTING!
  bool lighting = false;
  string val;
  //  if ( !window->get_gui_stringvar(window->id, "global-light", val ) )
  if ( ! window->getGui()->get( "global-light", val ) )
    cerr << "Huh? We can't access global lighting?!?!?!" << endl;
  else {
    if ( val == "1" )
      lighting = true;
    if ( lighting )
      //window->set_guivar( window->id, "global-light", "0" );
      window->getGui()->set( "global-light", "0" );
  }
  Log::log( MESSAGE, "*2* Forcing redraw!" );
  window->force_redraw();
  window->redraw_if_needed();
  Log::log( MESSAGE, "*6* Redraw completed..." );

  // Send a request to get the color and Z buffers, as well as the view
  // and GL matrices. Note - this assumes that nobody else draws into
  // the window between now and then, which may not be the case when we
  // have inertial rotation...
  FutureValue<GeometryData*> result("ZTexData");
  window->getData( GEOM_COLORBUFFER|GEOM_DEPTHBUFFER|GEOM_VIEW|GEOM_MATRICES,
		   &result );
  Log::log( MESSAGE, "*7* Getting data" );
  GeometryData *gd = result.receive();
  Log::log( MESSAGE, "*8* Data received" );
  
  // TURN LIGHTING BACK ON if needed
  if ( lighting )
    //window->set_guivar( window->id, "global-light", "1" );
    window->getGui()->set( "global-light", "1" );

  sendZTex( gd, eye, (ZTexRenderer *)request.groupInfo->group->getRenderer() );
  Log::log( MESSAGE, "*9* Sent ztex" );

  // Try 4-23
  delete gd;
}

void
ViewServerHelper::sendZTex( GeometryData * data, double * local_eye,
			    ZTexRenderer * ztexRenderer ) {
  Log::log( SemotusVisum::Logging::DEBUG, "In ViewServerHelper::sendZTex\n" );
  WallClockTimer fullTime;
  fullTime.start();
  int x = data->xres;
  int y = data->yres;
  double eye[3] = { data->view->eyep().x(),
		    data->view->eyep().y(),
		    data->view->eyep().z() };

  cerr << "================================" << endl;
  cerr << "Send ZTex:\n" << endl;
  cerr << "X/Y res: " << x << " " << y << endl;
  /*  cerr << "Colorbuffer size: " << data->colorbuffer->xres() << " " <<
    data->colorbuffer->yres() << endl;
  cerr << "Depthbuffer size: " << data->depthbuffer->xres() << " " <<
  data->depthbuffer->yres() << endl;*/
  int q;
  cerr << "Modelview matrix: " << endl;
  for ( q = 0; q < 16; q++ ) {
    if ( (q+1) % 4 == 0 ) cerr << endl;
    cerr << data->modelview[q] << " ";
  }
  cerr << "Projection matrix: " << endl;
  for ( q = 0; q < 16; q++ ) {
    if ( (q+1) % 4 == 0 ) cerr << endl;
    cerr << data->projection[q] << " ";
  }  
  cerr << "Viewport: " << data->viewport[0] << " " <<
    data->viewport[1] << " " << data->viewport[2] << " " <<
    data->viewport[3] << endl;
  cerr << "SCRun's Eyept: " << eye[0] << " " << eye[1] << " " << eye[2]
       << endl;
  cerr << "Our Eyept: " << local_eye[0] << " " << local_eye[1] << " "
       << local_eye[2] << endl;
  
  // Send the new znear and zfar parameters to all clients for this renderer.
  SetViewingMethod svm( false ); // Reply
    
  svm.setOkay( true, ZTexRenderer::name ); // Okay to switch
  // This really ought to be a geom...
  VMImageStreaming * vmi = scinew VMImageStreaming();
  vmi->setPerspective( data->view->fov(), data->znear, data->zfar );
  svm.setImageStream( vmi );
    
  svm.finish(); // Finish generating message
  // Send the message
  NetInterface::getInstance().
    sendPriorityDataToClients( ztexRenderer->getRenderGroup()->getClients(),
			       svm );
  // Convert float RGB colorbuffer to unsigned int RGB.
  Color c;
  char * image = scinew char[ x*y*3 ];
  for ( int ypos = 0; ypos < y; ypos++ )
    for ( int xpos = 0; xpos < x; xpos++ ) {
      c = data->colorbuffer->get_pixel( xpos, ypos );
      image[ (xpos + ypos*x)*3 ] = (char)(c.r() * 255);
      image[ (xpos + ypos*x)*3+1 ] = (char)(c.g() * 255);
      image[ (xpos + ypos*x)*3+2 ] = (char)(c.b() * 255);
    }

  // Feed the matrices and eyepoint to the renderer
  ztexRenderer->setMatrices( data->modelview, data->projection,
			     data->viewport );
  ztexRenderer->setEyepoint( local_eye );
  ztexRenderer->setDimensions( x, y );

  /* Send the render data */
  ztexRenderer->sendRenderData( image, x*y*3,
				(unsigned *)(data->depthbuffer),
				x * y * sizeof(unsigned int),
				true );
  delete image; // try 3-24

  fullTime.stop();
  char buffer[1000];
  snprintf( buffer, 1000, "Full ZTex Request took %0.2f seconds",
	    fullTime.time() );
  Log::log( SemotusVisum::Logging::MESSAGE, buffer );
  cerr << "End of ViewServerHelper::sendZTex" << endl;
}


ViewServer::ViewServer() : initialized( false ) {

  imageCount = 0;
  geomCount = 0;
  ztexCount = 0;
  dataWindowCount = 0;

  // We can't remove this callback, so we add it here...
  NetInterface::getInstance().onConnect( this, sendGroupViewers );

}

ViewServer::~ViewServer() {
  if ( initialized ) {
    /* FIXME - enabling this code causes aborts on Linux when SCIRun shuts
       down. Got to find some other way of cleaning up. A cleanup() function,
       perhaps?
    */
    //RenderRequest r;
    //r.groupInfo = NULL;
    //helper->sendRenderRequest( r );
    //helperThread->join();
    //delete helper;
  }
}

void
ViewServer::addRemoteModule( const string& id ) {
  cerr << "Add Module: " << id << endl;
  SemotusVisum::Message::Module *module =
    scinew SemotusVisum::Message::Module( ccast_unsafe(id), -1, -1,
					  SemotusVisum::Message::Module::ADD );
  remoteModules.add( module );
}
void
ViewServer::addRemoteModuleConnection( const string& from, const string& to ){
  cerr << "Connecting " << from << " to " << to << endl;
  for ( int i = 0; i < remoteModules.size(); i++ ) 
    if ( string(remoteModules[i]->getName()) == from ) {
      remoteModules[i]->addConnection( ccast_unsafe(to) );
      cerr << "Connected." << endl;
      return;
    }
  cerr << "Connection not found." << endl;
}

void
ViewServer::deleteRemoteModule( const string& id ) {
  cerr << "Delete Module: " << id << endl;
  for (int i = 0; i < remoteModules.size(); i++ )
    if ( !strcmp( remoteModules[i]->getName(), ccast_unsafe(id) ) ) {
      remoteModules.remove(i);
      cerr << "Deleted." << endl;
      return;
    }
  cerr << "Module not found." << endl;
}

void
ViewServer::deleteRemoteModuleConnection( const string& from,
					  const string& to) {
  cerr << "Disconnecting " << from << " to " << to << endl;
}

void
ViewServer::buildRemoteModules() {
  cerr << "Building list" << endl;
}

void
ViewServer::addDataViewWindow( ViewWindow * window ) {
  //cerr << "Adding data view window " << window << endl;

  char buffer[100];
  snprintf(buffer, 100, "Window%u", dataWindowCount );
  dataWindowCount++;
  dataWindows.add( scinew ViewWindowInfo( window, buffer ) );

  // New
  if ( dataWindows.size() != 1 ) {
    createRenderGroup( IMAGE, buffer );
    createRenderGroup( GEOM, buffer );
    createRenderGroup( ZTEX, buffer );
  }
  
  if ( dataWindows.size() == 1 ) {
    //cerr << "Starting up server" << endl;
    start();
  }
}

bool
ViewServer::existsDataViewWindow( ViewWindow * window ) {
  for ( int i = 0; i < dataWindows.size(); i++ )
    if ( dataWindows[i]->window == window )
      return true;
  return false;
}


  
void
ViewServer::removeDataViewWindow( ViewWindow * window ) {
  // FIXME - should also dissociate all render groups with this window.
  for ( int i = 0; i < dataWindows.size(); i++ )
    if ( dataWindows[i]->window == window ) {
      //cerr << "Removing data view window " << window << endl;
      dataWindows.remove(i);
      if ( dataWindows.size() == 0 ) {
	//cerr << "Stopping server" << endl;
	stop();
      }
      return;
    }
  Log::log( WARNING, "Removed data window does not exist in list!" );
}

void
ViewServer::start() {

  if ( !initialized )
    initialize();
  
  // Set up our callbacks
  addCallbacks();
  
  // Listen on network
  NetInterface::getInstance().listen();
}


void
ViewServer::stop() {

  // Stop listening on the interface
  NetInterface::getInstance().stop();

  // Remove all callbacks
  removeCallbacks();
}

void
ViewServer::initialize() {
  if ( initialized ) return;

  char buffer[ 1000 ];

  // Create helper thread
  helper       = scinew ViewServerHelper();
  helperThread = scinew Thread( helper, "ViewServerHelper" );
  // Detach helper here?
  helperThread->detach();

  
  /* Create default renderers and groups */
  snprintf( buffer, 1000, "%sDefault", ImageRenderer::name );
  RenderGroup * rg = scinew RenderGroup( buffer );
  RenderGroupInfo * rgi = scinew RenderGroupInfo( rg );
  imageGroups.add( rgi );

  snprintf( buffer, 1000, "%sDefault", GeometryRenderer::name );
  geomGroups.add( scinew RenderGroupInfo( scinew RenderGroup( buffer ) ) );
  snprintf( buffer, 1000, "%sDefault", ZTexRenderer::name );
  ztexGroups.add( scinew RenderGroupInfo( scinew RenderGroup( buffer ) ) );

  ImageRenderer * ir = scinew ImageRenderer;
  imageRenderers.add( ir );
  geomRenderers.add( scinew GeometryRenderer() );
  ztexRenderers.add( scinew ZTexRenderer() );

  /* Link the renderers and groups */
  imageGroups[0]->group->setRenderer( imageRenderers[0] );
  imageRenderers[0]->setRenderGroup( imageGroups[0]->group );
  geomGroups[0]->group->setRenderer( geomRenderers[0] );
  geomRenderers[0]->setRenderGroup( geomGroups[0]->group );
  ztexGroups[0]->group->setRenderer( ztexRenderers[0] );
  ztexRenderers[0]->setRenderGroup( ztexGroups[0]->group );

  if ( dataWindows.size() == 0 )
    Log::log( SemotusVisum::Logging::ERROR,
	      "Trying to init server without a window!" );
  else {
    // We have at least one window; set the renderers to this one.
    helper->addRenderWindowMap( imageGroups[0], dataWindows[0] );
    dataWindows[0]->window->getViewState( imageGroups[0]->state );
    helper->addRenderWindowMap( geomGroups[0], dataWindows[0] );
    dataWindows[0]->window->getViewState( geomGroups[0]->state );
    helper->addRenderWindowMap( ztexGroups[0], dataWindows[0] );
    dataWindows[0]->window->getViewState( ztexGroups[0]->state );
    Log::log( SemotusVisum::Logging::MESSAGE,
	      "Added default renderers to default view window" );
  }

  initialized = true;
}


void
ViewServer::addCallbacks() {

  // Set up all rendering callbacks
  int i;
  for ( i = 0; i < imageRenderers.size(); i++ )
    imageRenderers[i]->setCallbacks(); 
  for ( i = 0; i < geomRenderers.size(); i++ )
    geomRenderers[i]->setCallbacks();
  for ( i = 0; i < ztexRenderers.size(); i++ )
    ztexRenderers[i]->setCallbacks();
  
  // Add a callback for mousemove, as we want to be notified immediately
  mouseCallbackID = 
    SemotusVisum::Network::NetDispatchManager::getInstance().
    registerCallback( MOUSE_MOVE,
		      mouseCallback,
		      this,
		      true );
  
  // Set up our 'set viewing method' callback
  setViewingMethodCallbackID =
    SemotusVisum::Network::NetDispatchManager::getInstance().
    registerCallback( SET_VIEWING_METHOD,
		      setViewMethodCallback,
		      this,
		      true );

  // We also want to know when clients request ZTex objects.
  ztexCallbackID = 
    SemotusVisum::Network::NetDispatchManager::getInstance().
    registerCallback( GET_Z_TEX,
		      getZTexCallback,
		      this,
		      true );

  // Add callback for remote module viewing
  remoteModuleCallbackID =
    SemotusVisum::Network::NetDispatchManager::getInstance().
    registerCallback( XDISPLAY,
		      remoteModuleCallback,
		      this,
		      true );

}

void
ViewServer::removeCallbacks() {
  // Remove all callbacks for all renderers
  int i;
  for ( i = 0; i < imageRenderers.size(); i++ )
    imageRenderers[i]->removeCallbacks(); 
  for ( i = 0; i < geomRenderers.size(); i++ )
    geomRenderers[i]->removeCallbacks();
  for ( i = 0; i < ztexRenderers.size(); i++ )
    ztexRenderers[i]->removeCallbacks();

  // Remove server callbacks
  SemotusVisum::Network::NetDispatchManager::getInstance().
    deleteCallback( ztexCallbackID );
  SemotusVisum::Network::NetDispatchManager::getInstance().
    deleteCallback( setViewingMethodCallbackID );
  SemotusVisum::Network::NetDispatchManager::getInstance().
    deleteCallback( mouseCallbackID );
  SemotusVisum::Network::NetDispatchManager::getInstance().
    deleteCallback( remoteModuleCallbackID );
}

void
ViewServer::setViewMethodCallback( void * obj, MessageData *input ) {

  // Get our viewserver object
  ViewServer * vs = (ViewServer *)obj;

  // If we get NULL, return
  if ( vs == NULL || input == NULL )
    return;

  // Call a nonstatic callback function.
  vs->setViewingMethod( input );
}

void
ViewServer::setViewingMethod( MessageData *input ) {
  char buffer[1000];
  char * group = NULL;
  char * viewer = NULL;
  
  RenderGroupInfo * oldRenderGroup = getRenderGroupClient( input->clientName );
  
  SetViewingMethod * SetVM = (SetViewingMethod *)(input->message);
  /* If we switch to image streaming, reply and return true. */
  if ( !strcasecmp( SetVM->getMethod(),
		    ImageRenderer::name ) ) {
    cerr << "Image Streaming!" << endl;
    VMImageStreaming *vmi =
      SetVM->getImageStream();
    VMImageStreaming *vm = scinew VMImageStreaming;
    
    /* Check for lighting, shading, and fog */
    int lighting=-1, fog=-1;
    char * shading=NULL;
    int resX=-1, resY=-1, reduction=-1;
    int subimage=-1;
    
    // Get rendering parameters.
    if ( vmi ) vmi->getRendering( lighting, shading, fog );
    vm->setRendering( lighting, shading, fog );
    if ( vmi ) vmi->getResolution( resX, resY, reduction );
    vm->setResolution( resX, resY, reduction );
    if ( vmi ) vmi->getSubimage( subimage );
    if ( subimage != -1 )
      vmi->setSubimage( subimage ? true : false );
    
    SetViewingMethod svm( false ); // Reply
    svm.setImageStream( vm );
    
    if ( SetVM->getGroup() == NULL ) {
      snprintf( buffer, 1000, "%sDefault", ImageRenderer::name );
      // If there is no group specified, add it to the 'default' group.
      if ( addClientToRenderGroup( input->clientName, buffer ) ) {
	group = buffer;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       buffer );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "new" ) ) {
      cerr << "New group" << endl;
      if ( SetVM->getViewer() != NULL ) {
	viewer = SetVM->getViewer();
      }
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      
      char * groupName = createRenderGroup( IMAGE, viewer );
      
      if ( addClientToRenderGroup( input->clientName, groupName ) ) {
	group = groupName;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       group );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "standalone" ) ) {
      if ( SetVM->getViewer() != NULL )
	viewer = SetVM->getViewer();
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      char * groupName = createRenderGroup( IMAGE, viewer );
      if ( addClientToRenderGroup( input->clientName, groupName ) )
	group = groupName;
    }
    else {
      // Try adding it to the list. If that doesn't work, switch to the
      // default.
      snprintf( buffer, 1000, "%sDefault", ImageRenderer::name );
      if ( ! addClientToRenderGroup( input->clientName, SetVM->getGroup() ) ) {
	if ( addClientToRenderGroup( input->clientName, buffer ) )
	  group = buffer;
      }
      else 
	group = SetVM->getGroup();

      NetInterface::getInstance().modifyClientGroup( input->clientName,
						     group );
      
    }

    svm.setOkay( true, ImageRenderer::name,
		 group, viewer ); // Okay to switch
    svm.finish(); // Finish generating message

    
    // Send the message
    NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							  svm );
    cerr << "Sent svm to client " << input->clientName << endl;
    
    // Add a render request
    if ( group != NULL ) {
      //cerr << "Group name: " << group << endl;
      // FIXME - change shading params in info here...
      RenderGroupInfo * rgi = getRenderGroup( group );
      //cerr << "RGI: " << (void *)rgi << endl;

      /* Set subimaging in renderer */
      if ( rgi && subimage != -1 ) {
	cerr << "Setting subimage to " << (subimage ? true : false ) << endl;
	((ImageRenderer *)(rgi->group->getRenderer()))->setSubimage( subimage ?
								     true :
								     false );
      }
      RenderRequest r( rgi,
		       IMAGE);
      cerr << "Got render request" << endl;
      helper->sendRenderRequest( r );
    }
    cerr << "Sent render request" << endl;
  }
  else if ( !strcasecmp(((SetViewingMethod *)(input->message))->getMethod(),
			GeometryRenderer::name ) ) {

    if ( SetVM->getGroup() == NULL ) {
      snprintf( buffer, 1000, "%sDefault", GeometryRenderer::name );
      // If there is no group specified, add it to the 'default' group.
      if ( addClientToRenderGroup( input->clientName, buffer ) ) {
	group = buffer;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       buffer );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "new" ) ) {
      if ( SetVM->getViewer() != NULL )
	viewer = SetVM->getViewer();
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      char * groupName = createRenderGroup( GEOM, viewer );
      if ( addClientToRenderGroup( input->clientName, groupName ) ) {
	group = groupName;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       group );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "standalone" ) ) {
      if ( SetVM->getViewer() != NULL )
	viewer = SetVM->getViewer();
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      char * groupName = createRenderGroup( GEOM, viewer );
      if ( addClientToRenderGroup( input->clientName, groupName ) )
	group = groupName;
    }
    else {
      snprintf( buffer, 1000, "%sDefault", GeometryRenderer::name );
      // Try adding it to the list. If that doesn't work, switch to the
      // default.
      if ( ! addClientToRenderGroup( input->clientName, SetVM->getGroup() ) ) {
	if ( addClientToRenderGroup( input->clientName, buffer ) )
	  group = buffer;
      }
      else 
	group = SetVM->getGroup();
      
      NetInterface::getInstance().modifyClientGroup( input->clientName,
						     group );
     
    }
    SetViewingMethod svm( false ); // Reply
    
    svm.setOkay( true, GeometryRenderer::name,
		 group, viewer); // Okay to switch
    svm.finish(); // Finish generating message
    
    // Send the message
    NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							  svm );
    // Add a render request
    if ( group != NULL ) {
      RenderRequest r( getRenderGroup( group ),
		       GEOM );
      helper->sendRenderRequest( r );
    }
  }
  else if ( !strcasecmp(((SetViewingMethod *)(input->message))->getMethod(),
			ZTexRenderer::name ) ) {

    if ( SetVM->getGroup() == NULL ) {
      snprintf( buffer, 1000, "%sDefault", ZTexRenderer::name );
      // If there is no group specified, add it to the 'default' group.
      if ( addClientToRenderGroup( input->clientName, buffer ) ) {
	group = buffer;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       buffer );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "new" ) ) {
      if ( SetVM->getViewer() != NULL )
	viewer = SetVM->getViewer();
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      char * groupName = createRenderGroup( ZTEX, viewer );
      if ( addClientToRenderGroup( input->clientName, groupName ) ) {
	group = groupName;
	NetInterface::getInstance().modifyClientGroup( input->clientName,
						       group );
      }
    }
    else if ( !strcasecmp( SetVM->getGroup(), "standalone" ) ) {
      if ( SetVM->getViewer() != NULL )
	viewer = SetVM->getViewer();
      else 
	// Set a default. We always have at least one window...
	viewer = dataWindows[0]->name;
      char * groupName = createRenderGroup( ZTEX, viewer );
      if ( addClientToRenderGroup( input->clientName, groupName ) )
	group = groupName;
    }
    else {
      snprintf( buffer, 1000, "%sDefault", ZTexRenderer::name );
      // Try adding it to the list. If that doesn't work, switch to the
      // default.
      if ( ! addClientToRenderGroup( input->clientName, SetVM->getGroup() ) ) {
	if ( addClientToRenderGroup( input->clientName, buffer ) ) 
	  group = buffer;
      }
      else 
	group = SetVM->getGroup();
      
      NetInterface::getInstance().modifyClientGroup( input->clientName,
						     group );
    }

    SetViewingMethod svm( false ); // Reply
    
    svm.setOkay( true, ZTexRenderer::name,
		 group, viewer ); // Okay to switch
    svm.finish(); // Finish generating message


    // Send the message
    NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							  svm );
    
    // Add a render request for the view
    if ( group != NULL ) {
      RenderRequest r( getRenderGroup( group ),
		       VIEW,
		       NULL,
		       NULL,
		       input->clientName );
      helper->sendRenderRequest( r );
    }
    
  } 
  else {
    snprintf( buffer, 1000, "Duh, I don't know about renderer %s",
	      ((SetViewingMethod *)(input->message))->getMethod() );
    return;
  }

  // If we had an old group that isn't the same as our current group,
  // remove us from that group!
  if ( oldRenderGroup != NULL &&
       strcmp( oldRenderGroup->group->getName(), group ) ) {
    removeClientFromRenderGroup( input->clientName,
				 oldRenderGroup->group->getName() );
  }
}

void
ViewServer::mouseCallback( void * obj, MessageData *input ) {
  WallClockTimer mc;
  mc.start();
  
  // Get our viewserver object
  ViewServer * vs = (ViewServer *)obj;

  // If we get NULL, return
  if ( vs == NULL || input == NULL )
    return;

  // Call a nonstatic callback function.
  vs->localMouseCallback( input );
  mc.stop();
  
  char buffer[1000];
  snprintf( buffer, 1000, "Mouse callback took %0.2f seconds",
	    mc.time() );
  Log::log( SemotusVisum::Logging::MESSAGE, buffer );
}

void
ViewServer::localMouseCallback( MessageData * input ) {

  // Uh oh...this got called before the image renderer...
  Log::log( SemotusVisum::Logging::WARNING,
	    "Server got mouse event before renderer!" );
  MouseMove * mm = (MouseMove *)(input->message);
  int action;
  char button;
  int x, y;
  mm->getMove( x, y, button, action );
  if ( action == MouseMove::START ) action = MouseStart;
  else if ( action == MouseMove::DRAG ) action = MouseDrag;
  else if ( action == MouseMove::END ) action = MouseEnd;
  else // What is this? Punt...
    return;
  
  struct timeval eventTime;
  gettimeofday( &eventTime, NULL );
  int timeMS = eventTime.tv_sec * 1000 + eventTime.tv_usec / 1000;

  MouseEvent me;
  me.x = x; me.y = y; me.button = button; me.action = action;
  me.timeMS = timeMS;

  // Get the render group info from the client name - look in the image
  // renderers only.
  RenderGroupInfo * rgi = NULL;
  RenderGroup *rg = RenderGroup::getRenderGroup( input->clientName );
  for ( int i = 0; i < imageGroups.size(); i++ ) {
    if ( imageGroups[i]->group == rg ) {
      rgi = imageGroups[i];
      break;
    }
  }
  if ( rgi != NULL )
    helper->sendRenderRequest( RenderRequest( rgi,
					      MOUSE,
					      &me ) );

}

void
ViewServer::getZTexCallback( void * obj, MessageData *input ) {
  // Get our viewserver object
  ViewServer * vs = (ViewServer *)obj;

  // If we get NULL, return
  if ( vs == NULL || input == NULL )
    return;

  // Call a nonstatic callback function.
  vs->getZTex( input );

}

void
ViewServer::getZTex( MessageData * input ) {
  RenderGroupInfo * group =
    getRenderGroup(RenderGroup::getRenderGroup(input->clientName)->getName());
  helper->sendRenderRequest( RenderRequest( group,
					    ZTEX,
					    NULL,
					    (GetZTex *)(input->message),
					    NULL ) );
}

void
ViewServer::sendGroupViewers( void * obj, const char * clientName ) {
  if ( obj == NULL ) return;

  ((ViewServer *)obj)->sendGroupView( clientName );
}

void
ViewServer::sendGroupView( const char * clientName ) {
  int i;

  char buffer[1000];
  snprintf( buffer, 1000, "Sending group/viewer info to %s", clientName );
  Log::log( MESSAGE, buffer );
  
  GroupViewer gv( false );

  // Add all the group/window pairings
  for ( i = 0; i < imageGroups.size(); i++ ) {
    snprintf( buffer, 1000, "Image: %s->%s", imageGroups[i]->group->getName(),
	      helper->getWindow( imageGroups[i] )->name );
    Log::log( MESSAGE, buffer );
    gv.addGroup( imageGroups[i]->group->getName(),
		 helper->getWindow( imageGroups[i] )->name );
  }
  
  for ( i = 0; i < geomGroups.size(); i++ )
    gv.addGroup( geomGroups[i]->group->getName(),
		 helper->getWindow( geomGroups[i] )->name );

  for ( i = 0; i < ztexGroups.size(); i++ )
    gv.addGroup( ztexGroups[i]->group->getName(),
		 helper->getWindow( ztexGroups[i] )->name );

  gv.finish();
  NetInterface::getInstance().sendPriorityDataToClient( clientName,
							gv );
}

void
ViewServer::remoteModuleCallback( void *obj, MessageData *input ) {
  // Get our viewserver object
  ViewServer * vs = (ViewServer *)obj;

  // If we get NULL, return
  if ( vs == NULL || input == NULL )
    return;

  // Call a nonstatic callback function.
  vs->doRemoteModuleCallback( input );
}

char *
getBaseName( const char * fullName, int start, int length ) {
  int nstart = length-1;
  while ( fullName[nstart--] != '_' && nstart >= start );
  if ( nstart < start ) return NULL;
  while ( fullName[nstart--] != '_' && nstart >= start );
  if ( nstart < start ) return NULL; 
  nstart+=2;
  char * modName = scinew char[ length - nstart+1 ];
  memset( modName, 0, length - nstart+1 );
  strncpy( modName, fullName+nstart, length - nstart );
  return modName;
}

void
ViewServer::doRemoteModuleCallback( MessageData *input ) {
  // Create an XDisplay message
  XDisplay *x = (XDisplay *)(input->message);
  
  // If this is a refresh request, resend the modules
  if ( x->isRefreshRequest() ) {
    XDisplay outbound;
    network->read_lock();
    Module *module;
    SemotusVisum::Message::Module *m = NULL;
    int x,y;

    // Build a list of from-to connections.
    cerr << "We have " << network->nconnections() << " connections" << endl;
    Array1<string> from, to;
    string conn;
    int nstart = 0;
    for ( int j = 0; j < network->nconnections(); j++ ) {
      conn = network->connection( j )->id;
      cerr << "Breaking up connection " << conn << endl;
      char * con = ccast_unsafe( conn );
      char *from_=NULL, *to_=NULL;
      unsigned s;
      for ( s = 0; s < conn.length(); s++ ) {
	if ( con[s] == '_' && con[s+1] == 'p' ) break;
	if ( con[s] == '_' ) nstart = s;
      }
      nstart--;
      while ( con[nstart] != '_' ) nstart--;
      nstart++;
      cerr << "s = " << s << " nstart = " << nstart << endl;
      from_ = scinew char[s+1-nstart];
      memset( from_, 0, s+1-nstart );
      strncpy( from_, con+nstart, s-nstart );
      cerr << "From: " << from_ << endl;
      while ( con[s-3] != 't' && con[s-2] != 'o' ) s++;
      nstart = s;
      for ( ; s < conn.length(); s++ ) {
	if ( con[s] == '_' && con[s+1] == 'p' ) break;
	if ( con[s] == '_' ) nstart = s;
      }
      nstart--;
      while ( con[nstart] != '_' ) nstart--;
      nstart++;
      to_ = scinew char[ s + 1 - nstart ];
      memset( to_, 0, s+1-nstart );
      strncpy( to_, con+nstart, s-nstart );
      cerr << "To: " << to_ << endl;
      string fromString( from_ );
      string toString( to_ );
      from.add( fromString );
      to.add( toString );
    }

    // Build a module list
    cerr << "We have " << network->nmodules() << " modules!" << endl;
    for ( int i = 0; i < network->nmodules(); i++ ) {
      module = network->module(i);
      cerr << "Module full name: " << module->getID() << endl;
      char * conn = ccast_unsafe( module->getID() );
      char * modName = getBaseName( conn, 0, strlen( conn ) );
      string mName( modName );
      cerr << "Module small name: " << mName << endl;
      //cerr << "Adding module " << module->id << endl;
      module->getPosition( x, y );
      m =
	scinew SemotusVisum::Message::Module( //ccast_unsafe( module->id ),
					   ccast_unsafe(mName),
					   x, y,
					   SemotusVisum::Message::Module::ADD);
      
      for ( int k = 0; k < from.size(); k++ )
	if ( from[k] == /*module->id*/mName )
	  m->addConnection( ccast_unsafe( to[k] ) );
      
      outbound.addModule( m );
    }
      
    network->read_unlock();
    outbound.finish();
    NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							  outbound );
    
  }
  // Else
  else {
    char * display = x->getClientDisplay();
    char * module = x->getModuleName();
    cerr << "Module " << module << endl;
    cerr << "Display " << display << endl;

    if ( !display || !module ) {
      XDisplay xd;
      xd.setResponse( false, "Missing module/display!" );
      xd.finish();
      NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							    xd );
      return;
    }
    
    // Find the full module name;
    network->read_lock();
    string modName( module );
    char * name = NULL;
    char * baseName;
    for ( int i = 0; i < network->nmodules(); i++ ) {
      name = ccast_unsafe( network->module(i)->getID() );
      baseName = getBaseName( name, 0, strlen(name) );
      if ( !strcmp( module, baseName ) ) {
	cerr << "Got a match! Base name = " << baseName << ", fullname = " <<
	  name << endl;
	delete baseName;
	break;
      }
      else
	name = NULL;
      delete baseName;
    }
    network->read_unlock();
    
    if ( name == NULL ) {
      XDisplay xd;
      xd.setResponse( false, "Couldn't find module!" );
      xd.finish();
      NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							    xd );
    }
    else {
      string mod = name;//module;
      string dis = display;
      string command = mod + " initialize_ui " + dis;
      cerr << "Command: " << command << endl;
      
      // NOT YET - which TCL do we execute this under? TCL::execute( command );
      
      XDisplay xd;
      xd.setResponse( true );
      xd.finish();
      NetInterface::getInstance().sendPriorityDataToClient( input->clientName,
							    xd );
    }
  }
}

bool
ViewServer::addClientToRenderGroup( const char * client,
				    const char * group ) {
  RenderGroup * rg = getRenderGroup( group )->group;

  if ( rg == NULL ) {
    char buffer[1000];
    snprintf(buffer,1000,"No render group by the name of '%s'", group );
    Log::log( WARNING, buffer );
    return false;
  }

  return rg->addClient( client );

}

void
ViewServer::removeClientFromRenderGroup( const char * client,
					 const char * group ) {
  RenderGroup *rg = getRenderGroup( group )->group;

  if ( rg == NULL )
    return;

  cerr << "Removing client " << client << " from group " << rg->getName()
       << endl;
  rg->removeClient( client );
}

RenderGroupInfo *
ViewServer::getRenderGroup( const char * group ) {
  int i;

  // Image groups
  for ( i = 0; i < imageGroups.size(); i++ ) {
    cerr << "GetRenderGroup: Group = " << (void *)imageGroups[i]->group <<
      endl;
    if ( imageGroups[i]->group->getName() != NULL ) {
      cerr << "GetRenderGroup: Comparing " << group << " and " <<
	imageGroups[i]->group->getName() << endl;
      if ( !strcasecmp( imageGroups[i]->group->getName(), group ) ) {
	cerr << "Found it!" << endl;
	return imageGroups[i];
      }
    }
  }
  
  // Geom groups
  for ( i = 0; i < geomGroups.size(); i++ )
    if ( geomGroups[i]->group->getName() != NULL &&
	 !strcasecmp( geomGroups[i]->group->getName(), group ) )
      return geomGroups[i];
  

  // Ztex groups
  for ( i = 0; i < ztexGroups.size(); i++ )
    if ( ztexGroups[i]->group->getName() != NULL &&
	 !strcasecmp( ztexGroups[i]->group->getName(), group ) )
      return ztexGroups[i];

  return NULL; // Group not found
}

RenderGroupInfo *
ViewServer::getRenderGroupClient( const char * client ) {
  RenderGroup * rg = RenderGroup::getRenderGroup( client );
  int i;
  if ( rg == NULL )
    return NULL;

  // Image groups
  for ( i = 0; i < imageGroups.size(); i++ )
    if ( imageGroups[i]->group == rg )
      return imageGroups[i];

  // Geom groups
  for ( i = 0; i < geomGroups.size(); i++ )
    if ( geomGroups[i]->group == rg )
      return geomGroups[i];
  

  // Ztex groups
  for ( i = 0; i < ztexGroups.size(); i++ )
    if ( ztexGroups[i]->group == rg )
      return ztexGroups[i];
  
  return NULL; // Group not found
}

char *
ViewServer::createRenderGroup( int renderType, char * viewer ) {
  char buffer[300];
  RenderGroupInfo *group = NULL;
  int i;
  
  // First, find the viewer (data window)
  ViewWindowInfo *info = NULL;
  for ( i = 0; i < dataWindows.size(); i++ )
    if ( !strcasecmp( dataWindows[i]->name, viewer ) )
      info = dataWindows[i];

  // If the window doesn't exist
  if ( info == NULL ) {
    snprintf( buffer, 300,
	      "Trying to attach a new renderer to nonexistent window %s",
	      viewer );
    Log::log( SemotusVisum::Logging::ERROR, buffer );
    return NULL;
  }
  
  switch ( renderType ) {
  case IMAGE:
    {
      snprintf( buffer, 300, "%s%u", ImageRenderer::name, imageCount );
      imageCount++;
      group = scinew RenderGroupInfo( scinew RenderGroup( buffer ) );
info->window->getViewState( group->state );
      imageGroups.add( group );

      ImageRenderer * ir = scinew ImageRenderer();
      imageRenderers.add( ir );

      ir->setRenderGroup( group->group );
      group->group->setRenderer( ir );
      
      helper->addRenderWindowMap( group, info );
      
      return strdup( buffer );
    }
  case GEOM:
    {
      snprintf( buffer, 300, "%s%u", GeometryRenderer::name, geomCount );
      geomCount++;
      group = scinew RenderGroupInfo( scinew RenderGroup( buffer ) );
      info->window->getViewState( group->state );
      geomGroups.add( group );

      GeometryRenderer * gr = scinew GeometryRenderer();
      geomRenderers.add( gr );

      gr->setRenderGroup( group->group );
      group->group->setRenderer( gr );
      
      helper->addRenderWindowMap( group, info );
      
      return strdup( buffer );
    }
  case ZTEX:
    {
      snprintf( buffer, 300, "%s%u", ZTexRenderer::name, ztexCount );
      ztexCount++;
      group = scinew RenderGroupInfo( scinew RenderGroup( buffer ) );
      info->window->getViewState( group->state );
      ztexGroups.add( group );

      ZTexRenderer *zr = scinew ZTexRenderer();
      ztexRenderers.add( zr );

      zr->setRenderGroup( group->group );
      group->group->setRenderer( zr );

      helper->addRenderWindowMap( group, info );
      
      return strdup( buffer );
    }
  default:
    return NULL;
  }
}

void
ViewServer::destroyRenderGroup( const char * group ) {
  int i;

  // Image groups
  for ( i = 0; i < imageGroups.size(); i++ )
    if ( imageGroups[i]->group->getName() != NULL &&
	 !strcasecmp( imageGroups[i]->group->getName(), group ) ) {
      helper->removeRenderWindowMap( imageGroups[i],
				     helper->getWindow(imageGroups[i] ) );
      imageGroups.remove(i);
      return;
    }

  // Geom groups
  for ( i = 0; i < geomGroups.size(); i++ )
    if ( geomGroups[i]->group->getName() != NULL &&
	 !strcasecmp( geomGroups[i]->group->getName(), group ) ){
      helper->removeRenderWindowMap( geomGroups[i],
				     helper->getWindow(geomGroups[i] ) );
      geomGroups.remove(i);
      return;
    }
  
  // Ztex groups
  for ( i = 0; i < ztexGroups.size(); i++ )
    if ( ztexGroups[i]->group->getName() != NULL &&
	 !strcasecmp( ztexGroups[i]->group->getName(), group ) ){
      helper->removeRenderWindowMap( ztexGroups[i],
				     helper->getWindow(ztexGroups[i] ) );
      ztexGroups.remove(i);
      return;
    }
}

bool
ViewServer::needImage( ViewWindow * window ) {
  // FIXME - map calling window to whether or not an image renderer is
  // using it...
  window = NULL; // To make the compiler stop complaining...
  return true;
}

void
ViewServer::refreshView( const View &v,
			 RenderGroupInfo *group ) {

  Point eye = v.eyep();
  Point at = v.lookat();
  Vector up = v.up();
  
  SetViewingMethod svm( false ); // Reply
    
  svm.setOkay( true, group->group->getRenderer()->getName() );
  
  // This really ought to be a geom...
  VMImageStreaming * vmi = scinew VMImageStreaming();
  vmi->setEyePoint( eye.x(), eye.y(), eye.z() );
  vmi->setLookAtPoint( at.x(), at.y(), at.z() );
  vmi->setUpVector( up.x(), up.y(), up.z() );

  svm.setImageStream( vmi );
    
  svm.finish(); // Finish generating message


  // Send the message
  NetInterface::getInstance().
    sendPriorityDataToClients( group->group->getClients(),
			       svm );
}

void     
ViewServer::sendImage( char * image, int xres, int yres,
		       int offX, int offY, int fullX, int fullY,
		       Color background,
		       SemotusVisum::Rendering::Renderer *renderer ) {
  cerr << "In ViewServer::sendImage" << endl;
  
  unsigned char * testImage = (unsigned char *) image;
    int numColoredPixels = 0;
    for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
      if((unsigned int)testImage[ i ] != 0 || (unsigned int)testImage[ i+1 ] != 0 || (unsigned int)testImage[ i+2 ] != 0){
        //cerr << "<" << (unsigned int)testImage[ i ] << ", " << (unsigned int)testImage[ i+1 ] << ", " << (unsigned int)testImage[ i+2 ] << ">  ";
        numColoredPixels++;
      }
    }
    cerr << "**************************NUM COLORED PIXELS = " << numColoredPixels << endl;
 
  ImageRenderer * imageRenderer = (ImageRenderer *)renderer;
  if ( imageRenderer ) {
    char bkgd[3];
    bkgd[0] = (char)background.r();
    bkgd[1] = (char)background.g();
    bkgd[2] = (char)background.b();
    
 
    imageRenderer->sendRenderData( image,
				   xres*yres*3, /* RGB */
				   xres,
				   yres,
				   offX,
				   offY,
				   fullX,
				   fullY,
				   bkgd,
				   true );
    delete image;
  }
  cerr << "End of ViewServer::sendImage" << endl;
}


void
ViewServer::sendImage( char * image, int xres, int yres,
		       SemotusVisum::Rendering::Renderer *renderer ) {
  cerr << "In ViewServer::sendImage" << endl;

  // DEBUG CODE
  unsigned char * testImage = (unsigned char *) image;
    int numColoredPixels = 0;
    for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
      if((unsigned int)testImage[ i ] != 0 || (unsigned int)testImage[ i+1 ] != 0 || (unsigned int)testImage[ i+2 ] != 0){
        //cerr << "<" << (unsigned int)testImage[ i ] << ", " << (unsigned int)testImage[ i+1 ] << ", " << (unsigned int)testImage[ i+2 ] << ">  ";
        numColoredPixels++;
      }
    }
    //cerr << "**************************NUM COLORED PIXELS = " << numColoredPixels << endl;

  ImageRenderer * imageRenderer = (ImageRenderer *)renderer;
  if ( imageRenderer ) {
    // Flip the image. Stupid OpenGL/Java using opposite y axes!
    // Disabled for testing with C++ client!
#if 0
    WallClockTimer wct;
    wct.start();
    char * tmp = scinew char[xres*3];
    for ( int k = 0; k < yres/2; k++ ) {
      memcpy( tmp, image+k*(xres*3), xres*3 );
      memcpy( image+k*(xres*3), image+(yres-k-1)*(xres*3), xres*3 );
      memcpy( image+(yres-k-1)*(xres*3), tmp, xres*3 );
    }
    delete tmp;
    wct.stop();
    cerr << "Flipping took " << wct.time() << " ms " << endl;
#endif

    // DEBUG CODE
    unsigned char * testImage = (unsigned char *) image;
    int numColoredPixels = 0;
    for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
      if((unsigned int)testImage[ i ] != 0 || (unsigned int)testImage[ i+1 ] != 0 || (unsigned int)testImage[ i+2 ] != 0){
        //cerr << "<" << (unsigned int)testImage[ i ] << ", " << (unsigned int)testImage[ i+1 ] << ", " << (unsigned int)testImage[ i+2 ] << ">  ";
        numColoredPixels++;
      }
    }
    //cerr << "**************************NUM COLORED PIXELS = " << numColoredPixels << endl;
   
    
    imageRenderer->sendRenderData( image,
				   xres*yres*3, /* RGB */
				   xres,
				   yres,
				   true );
  }
  delete image;

  cerr << "End of ViewServer::sendImage" << endl;
}

} // End namespace SCIRun
