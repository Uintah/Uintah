/*
 *
 * uiHelper: User interface helper - interfaces with rest of client.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <UI/uiHelper.h>
#include <UI/MiscUI.h>
#include <UI/UserInterface.h>
#include <Network/NetDispatchManager.h>
#include <Rendering/Renderer.h>
#include <Message/XDisplay.h>

#include <Util/ClientProperties.h>

namespace SemotusVisum {


uiHelper::uiHelper() : drawid(""), annotationMode(-1) {
  gvID =
    NetDispatchManager::getInstance().registerCallback( GROUP_VIEWER,
							__groupViewer,
							this,
							true );
  gclID =
    NetDispatchManager::getInstance().registerCallback( GET_CLIENT_LIST,
							__getClientList,
							this,
							true );
  collID =
    NetDispatchManager::getInstance().registerCallback( COLLABORATE,
							__collaborate,
							this,
							true );
  chatID =
    NetDispatchManager::getInstance().registerCallback( CHAT,
							__chat,
							this,
							true );

  xID =
    NetDispatchManager::getInstance().registerCallback( XDISPLAY,
							__xdisplay,
							this,
							true );
}


uiHelper::~uiHelper() {
  NetDispatchManager::getInstance().deleteCallback( gvID );
  NetDispatchManager::getInstance().deleteCallback( gclID );
  NetDispatchManager::getInstance().deleteCallback( collID );
  NetDispatchManager::getInstance().deleteCallback( chatID );
  NetDispatchManager::getInstance().deleteCallback( xID );
  
}

void 
uiHelper::groupViewer( MessageData * message ) {
  GroupViewer *gv = (GroupViewer *)(message->message);

  /* Adding a group? */
  if ( gv->isListAdd() ) {
    Log::log( DEBUG, "Adding group " + gv->getGroupName() +
	      ", viewer " + gv->getGroupViewer() );
    groupViewers.push_back( groupListItem( gv->getGroupName(),
					   gv->getGroupViewer() ) );

    // Update the UI
    string result;
    if ( eval("updateGroupViewer add " + gv->getGroupName() + " " +
	      gv->getGroupViewer(), result ) != TCL_OK ) {
      cerr << "Error updating group/viewer: " + result << endl;
    }
  }
  /* Subtracting a group? */
  else if ( gv->isListSub() ) {
    Log::log( DEBUG, "Subtracting group " + gv->getGroupName() +
	      ", viewer " + gv->getGroupViewer() );
    vector<groupListItem>::iterator i;
    groupListItem gli( gv->getGroupName(), gv->getGroupViewer() );
    for ( i = groupViewers.begin();
	  i != groupViewers.end();
	  i++ ) {
      if ( i->equals( gli ) ) {
	groupViewers.erase( i );
	
	// Update the UI
	string result;
	if ( eval("updateGroupViewer sub " + gv->getGroupName() + " " +
		  gv->getGroupViewer(), result ) != TCL_OK ) {
	  cerr << "Error updating group/viewer: " + result << endl;
	}
	return;
      }
    }
    Log::log( WARNING, "Group/viewer not found!" );
  }
  /* Filling the list from scratch? */
  else if ( gv->isListFill() ) {
    Log::log( DEBUG, "Filling group/viewer list with " +
	      mkString( gv->getGroupNames().size() ) + " pairs");
    groupViewers.clear();
    groupViewers = gv->getGroupNames();

    // Update the UI
    string result, command;
    command = "updateGroupViewer fill ";
    for ( unsigned i = 0; i < groupViewers.size(); i++ ) {
      command += "\"" + groupViewers[i].group + "\" \"" + groupViewers[i].viewer + "\" ";
    }
    if ( eval( command, result ) != TCL_OK ) {
      cerr << "Error updating group/viewer: " + result << endl;
    }
  }
  else {
    Log::log( ERROR, "Group viewer was not an add, subtract, or fill!" );
    return;
  }
  
}

void 
uiHelper::getClientList( MessageData * message ) {
  GetClientList * gcl = ( GetClientList * )(message->message);

  /* Adding a client? */
  if ( gcl->isListAdd() ) {
    Log::log( DEBUG, "Adding client " + gcl->getClientName() +
	      " at address " + gcl->getClientAddr() +
	      " with group " + gcl->getClientGroup() );
    clientAddrs.push_back( clientAddrGroup( gcl->getClientName(),
					    gcl->getClientAddr(),
					    gcl->getClientGroup() ) );
    // Update the UI
    string result;
    if ( eval("updateClientList add \"" + gcl->getClientName() + "\" \"" +
	      gcl->getClientAddr() + "\" \"" + gcl->getClientGroup() + "\"",
	      result ) != TCL_OK ) {
      cerr << "Error updating client: " + result << endl;
    }
  }
  /* Subtracting a client? */
  else if ( gcl->isListSub() ) {
    Log::log( DEBUG, "Subtracting client " + gcl->getClientName() +
	      ", address " + gcl->getClientAddr() );
    vector<clientAddrGroup>::iterator i;
    clientAddrGroup gli( gcl->getClientName(),
			 gcl->getClientAddr(),
			 gcl->getClientGroup() );
    for ( i = clientAddrs.begin();
	  i != clientAddrs.end();
	  i++ ) {
      if ( i->equals( gli ) ) {
	clientAddrs.erase( i );

	// Update the UI
	string result;
	if ( eval("updateClientList sub \"" + gcl->getClientName() + "\" \"" + 
		  gcl->getClientAddr() + "\" \"" + gcl->getClientGroup() +
		  "\"",
		  result ) != TCL_OK ) {
	  cerr << "Error updating client: " + result << endl;
	}
	return;
      }
    }
    Log::log( WARNING, "Client/group not found!" );
  }
  /* Modifying a client? */
  else if ( gcl->isListModify() ) {
    Log::log( DEBUG, "Modifying client " + gcl->getClientName() +
	      ", address " + gcl->getClientAddr() + " to group " +
	      gcl->getClientGroup() );
    vector<clientAddrGroup>::iterator i;
    clientAddrGroup gli( gcl->getClientName(),
			 gcl->getClientAddr(),
			 gcl->getClientGroup() );
    for ( i = clientAddrs.begin();
	  i != clientAddrs.end();
	  i++ ) {
      if ( i->equals( gli ) ) {
	i->group = gcl->getClientGroup();

	// Update the UI
	//theUI->modifyClient( *i );
	Log::log( DEBUG, "Updating client "+ gcl->getClientGroup() );
	string result;
	if ( eval("updateClientList modify \"" + gcl->getClientName() +
		  "\" \"" + gcl->getClientAddr() + "\" \"" +
		  gcl->getClientGroup() + "\"",
		  result ) != TCL_OK ) {
	  cerr << "Error updating client: " + result << endl;
	}
	Log::log( DEBUG, "Client update done" );
	return;
      }
    }
    Log::log( WARNING, "Client/group not found!" );
  }
  /* Filling the list from scratch? */
  else if ( gcl->isListFill() ) {
    Log::log( DEBUG, "Filling client/addr list with " +
	      mkString( gcl->getClientNames().size() / 3 ) + " triplets");
    clientAddrs.clear();
    vector<string> &names = gcl->getClientNames();
    for ( unsigned i = 0; i < names.size(); i+=3 )
      clientAddrs.push_back( clientAddrGroup( names[ i ],
					      names[ i+1 ],
					      names[ i+2 ] ) );
    
    // Update the UI
    string result, command;
    command = "updateClientList fill ";
    for ( unsigned i = 0; i < clientAddrs.size(); i++ )
      command += "\"" + clientAddrs[i].client + "\" \"" + clientAddrs[i].addr +
	"\" \"" + clientAddrs[i].group + "\"";
    if ( eval( command, result ) != TCL_OK ) {
      cerr << "Error updating client: " + result << endl;
    }
  }
  else {
    Log::log( ERROR, "Client List was not an add, subtract, mod, or fill!" );
  }
}

void  
uiHelper::grabZTex( const double * matrix ) {
  GetZTex * gzt = scinew GetZTex();
  
  gzt->setTransform( matrix );
  gzt->finish();
  sendMessageToServer( gzt );
}

void   
uiHelper::grabZTex( const Point3d eye, const Point3d at, const Vector3d up) {
  GetZTex * gzt = scinew GetZTex();

  gzt->setEyePoint( eye.x, eye.y, eye.z );
  gzt->setAtPoint( at.x, at.y, at.z );
  gzt->setUpVector( up.z, up.y, up.z );
  gzt->finish();
  sendMessageToServer( gzt );
}


void
uiHelper::collaborate( MessageData * message ) {

  // No renderer? Ignore the collaboration message
  if ( UserInterface::renderer() == NULL )
    return;
  
  Collaborate *c = (Collaborate *)(message->message);

  // Get the annotation data
  int i;
  
  // Pointers
  PointerData *p = NULL;
  for ( i = 0; i < c->numPointers(); i++ ) {
    p = c->getPointer(i);
    Log::log( DEBUG, "Adding remote pointer: " + p->output() );
    UserInterface::renderer()->addPointer( Point3d( p->x, p->y, p->z),
				 p->theta,
				 p->ID,
				 p->width,
				 p->color );
  }
  
  // Text
  TextData *t = NULL;
  for ( i = 0; i < c->numText(); i++ ) {
    t = c->getText(i);
    Log::log( DEBUG, "Adding remote text: " + t->output() );
    UserInterface::renderer()->addText( Point3d( t->x, t->y, 0.0 ),
			      t->_text, t->ID,
			      t->size, t->color );
  }
  
  // Drawing
  DrawingData *d = NULL;
  int j;
  Point3d point;
  for ( i = 0; i < c->numDrawings(); i++ ) {
    d = c->getDrawing(i);
    Log::log( DEBUG, "Adding remote drawing: " + d->output() );
    string name = UserInterface::renderer()->newDrawing(d->ID, d->width, d->color );
    for ( j = 0; j < d->numSegments(); j++ ) {
      point = d->getSegment( j );
      UserInterface::renderer()->addDrawingSegment( point, name);
    }
  }
}

void
uiHelper::chat( MessageData * message ) {
  Chat * c = (Chat *)(message->message);

  addTextToChatWindow( c->getName(), c->getText() );
}

void
uiHelper::Xdisplay( MessageData * message ) {
  XDisplay * x = (XDisplay *)(message->message);

  UserInterface::lock();
  if ( x->isModuleSetup() ) {
    vector<Module>& modlist = x->getModules();
    for (unsigned i = 0; i < modlist.size(); i++ ) {
      string command;
      if ( modlist[i].isRemoved() ) {

      }
      else if ( modlist[i].isModification() ) {
      }
      else {
	command = "addModule ";
	command += modlist[i].getName() + " ";
	command += mkString(modlist[i].getX()) + " ";
	command += mkString(modlist[i].getY()) + " ";
	vector<string> &conns = modlist[i].getConnections();
	for ( unsigned j = 0; j < conns.size(); j++ )
	  command += conns[j] + " ";
      }
      // Debug - delme!
      cerr << "Command = -" << command << "-" << endl;
      execute(command);
    }
    execute("rebuildModuleConnections");
  }
  else if ( x->isDisplayResponse() ) {
    if ( x->isResponseOkay() ) {
      Log::log( DEBUG, "Request to view module is okay!" );
    }
    else {
      execute("showmsg error Unable to display remote module: " +
	      x->getErrorText());
    }
  }
  UserInterface::unlock();
}


/**
 * Adds text to the chat window locally; ie, it interacts with the 
 * local user interface.
 *
 * @param    client         Name of the client adding the text.
 * @param    text           Text to add.
 */
void  
uiHelper::addTextToChatWindow( const string client, const string text ) {
  Log::log( DEBUG, "Adding text to chat window" );

  // Add the new text
  string result;
  if ( eval( "addchat " + client + ":  " + text, result ) != TCL_OK ) {
    cerr << "Error adding chat text: " + result << endl;
  }
}


    
void 
uiHelper::sendChat( const string chat ) {

  // Add it locally
  addTextToChatWindow( "local", chat );
  
  // Send it to the server
  Log::log( DEBUG, "Sending chat " + chat + " to server" );
  Chat * c = scinew Chat();

  c->setText( chat );
  c->finish();
  NetInterface::getInstance().sendDataToServer( c );
}

void  
uiHelper::viewAnnotations() {
  if ( UserInterface::renderer() == NULL ) {
    Log::log( MESSAGE, "No annotations to view!" );
    return;
  }
  
  vector<string> items = UserInterface::renderer()->getInfo();

  /*cerr << "===================\nAnnotations" << endl;
    for ( unsigned i = 0; i < items.size(); i++ )
    cerr << items[i] << endl;
  */

  string result;
  string command = "viewAnnotations ";
  for ( unsigned i = 0; i < items.size(); i++ )
    command += items[i] + " ";
  if ( eval( command, result ) != TCL_OK ) {
    cerr << "FIXME - use exec instead of eval?" << endl;
    cerr << "Error viewing annotations: " + result << endl;
  }
  
}

void
uiHelper::updateProperties() {
  cerr << "Updating properties!" << endl;

  
  string result, command;
  command = "setCompressors";
  for ( unsigned i = 0; i < ClientProperties::compressionFormats.size(); i++ )
    command += " " + ClientProperties::compressionFormats[i];
  if ( eval( command, result ) != TCL_OK ) {
    cerr << "Error updating compressors: " + result << endl;
  }

  command = "setRenderers";
  for ( unsigned i = 0; i < ClientProperties::serverRenderers.size(); i++ )
    command += " \"" + ClientProperties::serverRenderers[i] + "\"";
  if ( eval( command, result ) != TCL_OK ) {
    cerr << "Error updating renderers: " + result << endl;
  }

  command = "setTransferModes";
  for ( unsigned i = 0; i < ClientProperties::transferModes.size(); i++ )
    command += " " + ClientProperties::transferModes[i];
  if ( eval( command, result ) != TCL_OK ) {
    cerr << "Error updating transfer modes: " + result << endl;
  }
}

void
uiHelper::setFPS( const double fps ) {
  
  // Round to nearest tenth of a frame per second.
  int fps_whole = (int)fps;
  int fps_tenths = (int)((fps-fps_whole)*10);

  string command = "setFPS " + mkString( fps_whole ) + "." +
    mkString( fps_tenths );
  
  /* FIXME - lock UI here? */
  execute(command);
}

void
uiHelper::setCompression( const string &mode ) {
  execute("setCompression " + mode );
}

void
uiHelper::setTransfer( const string &mode ) {
  execute("setTransfer " + mode );
}

void
uiHelper::enableAnnotation( const string name, const bool enable,
			    void * obj ) {
#if 0
  cerr << "In void uiHelper::enableAnnotation" << endl;
  if ( obj == NULL )
    return;
  uiHelper *helper = (uiHelper *)obj;
  
  if ( helper->theUI->renderer == NULL )
    return;

  helper->theUI->renderer->enableAnnotation( name, enable );
  cerr << "In uiHelper::enableAnnotation, calling redraw, thread id is " << pthread_self() << endl;
  glutUI::getInstance().redraw();
#endif
}

void
uiHelper::deleteAnnotation( const string name, void * obj ) {
#if 0
  if ( obj == NULL )
    return;
  uiHelper *helper = (uiHelper *)obj;

  if ( helper->theUI->renderer == NULL )
    return;

  helper->theUI->renderer->deleteAnnotation( name );
  cerr << "In uiHelper::deleteAnnotation, calling redraw, thread id is " << pthread_self() << endl;
  glutUI::getInstance().redraw();
#endif
}


}
