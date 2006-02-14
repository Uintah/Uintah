#include <UI/SVCallback.h>
#include <UI/uiHelper.h>
#include <UI/OpenGL.h>
#include <UI/GuiArgs.h>
#include <UI/UserInterface.h>

#include <Thread/Thread.h>
#include <Rendering/Renderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>

#include <errno.h>

namespace SemotusVisum {

void
SVCallback::tcl_command(GuiArgs &args, void * data) {
  uiHelper *helper   = UserInterface::getHelper();
  Renderer *renderer = UserInterface::renderer();
  
  if(args.count() < 2){
    args.error("ui needs a minor command");
    return;
  }
  if(args[1] == "quit"){
    cerr << "Exiting... " << endl;
    Thread::exitAll(0);
  }
  else if(args[1] == "connect"){
    int port = atoi( args[4].c_str() );
    if ( NetInterface::getInstance().connectToServer( args[3], port ) ) {
      execute("setConnection connect \"" + args[3] + ":" + args[4] + "\"" ); 
    }
    else {
      execute("showmsg error Cannot connect to server " + args[3] + ":" +
	      mkString(port) + " : " + strerror(errno) );
    }
  }
  else if (args[1] == "disconnect") {
    NetInterface::getInstance().disconnectFromServer();
    execute("setConnection disconnect foobar");
  }
  else if (args[1] == "remoteModule") {
    if ( args[2] == "refresh" ) {
      XDisplay *x = new XDisplay;
      x->setRefreshRequest( true );
      x->finish();
      helper->sendMessageToServer( x );
    }
    else if (args[2] == "request" ) {
      Log::log( DEBUG, "Requesting remote module " + args[3] );
      XDisplay *x = new XDisplay;
      x->setDisplay( "" );
      x->setModuleName( args[3] );
      x->finish();
      helper->sendMessageToServer( x );
    }
  }
  else if (args[1] == "annotateMode" ) {
    if ( !renderer )
      return;
    if ( args[2] == "pointer" )
      helper->setAnnotationMode(CollaborationItem::POINTER);
    else if ( args[2] == "text" )
      helper->setAnnotationMode( CollaborationItem::TEXT );
    else if ( args[2] == "draw" )
      helper->setAnnotationMode(CollaborationItem::DRAWING);
    else if ( args[2] == "off" )
      helper->setAnnotationMode( -1 );
  }
  else if (args[1] == "setLighting" ) {
    if ( args[2] == "off" && renderer ) {
      renderer->setLighting( false );
    }
    else if ( args[2] == "on" && renderer ) {
      renderer->setLighting( true );
    }
  }
  else if (args[1] == "setlight") {
    if ( !renderer )
      return;
    int light = atoi( args[2] ) - 1;
    bool on =  args[3] == "1" ? true : false;
    renderer->setLight( light, on );
  }
  else if (args[1] == "setShading") {
    if ( !renderer )
      return;
    if ( args[2] == "flat" )
      renderer->setShadingType( Renderer::SHADING_FLAT );
    else if ( args[2] == "gouraud" )
      renderer->setShadingType( Renderer::SHADING_GOURAUD );
    else if ( args[2] == "wireframe" )
      renderer->setShadingType( Renderer::SHADING_WIREFRAME );
    else
      cerr << "Unknown shading type: " << args[2] << endl;
  }
  else if (args[1] == "setFog") {
    if ( !renderer )
      return;
    bool on = args[2] == "1" ? true : false;
    renderer->setFog( on );
  }
  else if (args[1] == "clientlistrefresh") {
    GetClientList * gcl = new GetClientList();
    gcl->finish();
    helper->sendMessageToServer( gcl );
  }
  else if (args[1] == "addchat") {
    cerr << "FIXME - add to ui helper, too!" << endl;
    if ( args.count() > 2 ) 
      cerr << "Got chat: " << args[2] << endl;
  }
  else if (args[1] == "changeCompression") {
    
    Compression *c = new Compression();
    c->setCompressionType( args[2] );
    c->finish();
    helper->sendMessageToServer( c );
  }
  else if (args[1] == "changeTransfer") {
    Log::log( DEBUG, "Changing transfer mode to " + args[2] );
    
    Transfer *t = new Transfer();
    t->setTransferType( args[2] );
    t->finish();
    helper->sendMessageToServer( t );
  }
  else if (args[1] == "changeGroup") {
    groupListItem gv( args[2], args[3] );
    helper->sendGroupToServer( gv, args[4] );
  }
  else if (args[1] == "listvisuals") {
    OpenGL::listvisuals( args );
  }
  else if (args[1] == "setvisual") {
    int idx = atoi( args[3] );
    int width = atoi( args[4] );
    int height = atoi( args[5] );
    
    OpenGL::setvisual( args[2], idx, width, height );
  }
  else if (args[1] == "redraw") {
    //cerr << "In SVCallback::tcl_command, args = redraw, calling draw, thread id is " << pthread_self() << endl;
    draw();
  }
  else if (args[1] == "getAnnotations" ) {
    if ( renderer == NULL )
      return;
    vector<string> items = renderer->getInfo();
    for ( unsigned i = 0; i < items.size(); i++ )
      Tcl_AppendElement( UserInterface::getInterp(),
			 items[i].c_str() );
  }
  else if (args[1] == "enableAnnotation") {
    if ( renderer == NULL )
      return;
    
    string item = args[2];
    bool enable = (args[3] == "1");
    
    renderer->enableAnnotation( item, enable );
    //cerr << "In SVCallback::tcl_command, args = enableAnnotation, calling draw, thread id is " << pthread_self() << endl;
    draw();
  }
  else if (args[1] == "deleteAnnotation") {
    if ( renderer == NULL )
      return;

    string item = args[2];
    renderer->deleteAnnotation( item );
    //cerr << "In SVCallback::tcl_command, args = deleteAnnotation, calling draw, thread id is " << pthread_self() << endl;
    draw();
  }
  else if ( args[1] == "mtranslate"){
    if ( renderer != NULL ) {
      switch (helper->getAnnotationMode()) {
      case CollaborationItem::POINTER:
	if ( args[2] == "start" ) {
	  helper->startx = atoi( args[3] );
	  helper->starty = atoi( args[4] );
	  helper->starty = renderer->window_height - helper->starty; // Flip y
	}
	else if ( args[2] == "end" ) {
	  int x = atoi( args[3] );
	  int y = renderer->window_height - atoi( args[4] ); // Flip y
	  double theta = 0;
	  if ( x != helper->startx )
	    theta = (int) (atan2( (double) (y-helper->starty) , (double)(x-helper->startx) ));
	  Point3d pt( helper->startx, helper->starty, 0);
	  renderer->addPointer( pt, theta );
          //cerr << "In SVCallback::tcl_command, args = mtranslate, calling draw, thread id is " << pthread_self() << endl;
	  draw();
	}
	break;
      case CollaborationItem::TEXT:
	if ( args[2] == "end" ) {
	  // Only place text when we release the button
	  string result;
	  eval("getAnnotateText", result);
	  
	  if ( result != "" ) {
	    int x = atoi(args[3]);
	    int y = renderer->window_height - atoi(args[4]); // Flip y
	    Point3d pt( x, y, 0 );
	    renderer->addText( pt, result );
            //cerr << "In SVCallback::tcl_command, args = mtranslate, calling draw, thread id is " << pthread_self() << endl;
	    draw();
	  }
	}
	break;
      case CollaborationItem::DRAWING:
	if ( args[2] == "end" ) {
	  // Only act when we release the button
	  bool doRedraw = true;

	  int x = atoi( args[3] );
	  int y = renderer->window_height - atoi( args[4] ); // Flip y

	  if ( helper->drawid == "" ) {
	    helper->drawid = renderer->newDrawing();
	    doRedraw = false; // First point doesn't get drawn
	  }
	  Point3d pt( x, y, 0 );
	  renderer->addDrawingSegment( pt, helper->drawid );
	  if ( doRedraw )
            //cerr << "In SVCallback::tcl_command, args - mtranslate, calling draw, thread id is " << pthread_self() << endl;
	    draw();
	}
	break;
      default:
	Log::log( DEBUG, "Translate!" );
	MouseEvent me = mouseEvent( args );
	me.button = 'L';
	renderer->mouseMove( me );
      }
    }
  }
  else if(args[1] == "mrotate"){
    if ( renderer != NULL ) {
      switch (helper->getAnnotationMode()) {
      case CollaborationItem::POINTER:/* No - op */
	break;
      case CollaborationItem::TEXT:/* No - op */
	break;
      case CollaborationItem::DRAWING:
	if ( args[2] == "end" ) {
	  // Cancel!
	  renderer->deleteDrawing( helper->drawid );
	  helper->drawid = "";
          //cerr << "In SVCallback::tcl_command, args = mrotate, calling draw, thread id is " << pthread_self() << endl;
	  draw();
	}
	break;
      default:
	Log::log( DEBUG, "Rotate!" );
	MouseEvent me = mouseEvent( args );
	me.button = 'M';
	renderer->mouseMove( me );
      }
    }
  }
  else if(args[1] == "mscale"){
    if ( renderer != NULL ) {
      switch (helper->getAnnotationMode()) {
      case CollaborationItem::POINTER:/* No - op */
	break;
      case CollaborationItem::TEXT:/* No - op */
	break;
      case CollaborationItem::DRAWING:
	if ( args[2] == "end" ) {
	  // Finish our drawing
	  renderer->finishDrawing( helper->drawid );
	  helper->drawid = "";
          //cerr << "In SVCallback::tcl_command, args = mscale, calling draw, thread id is " << pthread_self() << endl;
	  draw();
	}
	break;
      default:
	Log::log( DEBUG, "Scale!" );
	MouseEvent me = mouseEvent( args );
	me.button = 'R';
	renderer->mouseMove( me );
      }
    }
  }
  else if (args[1] == "gohome") {
    ((GeometryRenderer *)renderer)->homeView();
  }
  else if (args[1] == "getZTex") {
    ZTexRenderer *z = (ZTexRenderer *)renderer;
    cerr << "Doing getztex." << endl;
    cerr << "Eye = " << z->getEye().toString() << endl;
    cerr << "At = " << z->getAt().toString() << endl;
    cerr << "Up = " << z->getUp().toString() << endl;
    helper->grabZTex( z->getEye(), z->getAt(), z->getUp() );
    cerr << "Grabbed!" << endl;
  }
  else {
    cerr << "Unknown command -" << args[1] << "-" << endl;
  }
}

MouseEvent
SVCallback::mouseEvent( GuiArgs &args ) {
  MouseEvent me;

  if ( args[2] == "start"){
    me.action=START;
  } else if(args[2] == "end"){
    me.action=END;
  } else if(args[2] == "move"){
    me.action=DRAG;
  } else {
    args.error("Unknown mouse action");
    return me;
  }

  me.x = atoi( args[3] );
  me.y = atoi( args[4] );

  int state=0;
  int btn=0;
  if(args.count() == 7){
    state = atoi( args[5] );
    btn = atoi( args[6] );
  }
  int time=0;
  if(args.count() == 8){
    time = atoi( args[7] );
  }
  if(args.count() == 6){
    time = atoi( args[5] );
  }

  return me;
}


}
