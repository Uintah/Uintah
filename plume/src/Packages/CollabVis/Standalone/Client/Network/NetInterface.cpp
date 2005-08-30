/*
 *
 * etInterface: Provides access to the network
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#include <unistd.h>
#include <errno.h>
#include <Network/NetInterface.h>
#include <Malloc/Allocator.h>
#include <Util/ClientProperties.h>
#include <UI/UserInterface.h>
#include <UI/uiHelper.h>
#include <Logging/Log.h>

using namespace std;

namespace SemotusVisum {

using namespace SCIRun;

/* The singleton Network Interface */
NetInterface
NetInterface::net;

/* Data marker for the end of XML data and the beginning of raw data. */
const char * const
NetInterface::dataMarker = "\001\002\003\004\005";


NetInterface::NetInterface() : multicastEnabled( true ),
			       serverConnection( NULL ),
			       multicastConnection( NULL ) {
  //Log::log( DEBUG, "[NetInterface::NetInterface] entered" ); //, thread id = " + mkString((int) pthread_self()) );
  //Log::log( DEBUG, "[NetInterface::NetInterface] leaving, thread id = " + mkString((int) pthread_self()) );
}

NetInterface::~NetInterface() {
  // This is only called on program shutdown!
}

void
NetInterface::connectToMulticast( string group, int port ) {
  Log::log( ENTER, "[NetInterface::connectToMulticast] entered, thread id = " + mkString((int) pthread_self()) );

  cerr << "NetInterface::connectToMulticast - group = " << group << ", port = " << port << endl;
  Log::log( MESSAGE, 
	    "[NetInterface::connectToMulticast] Connecting to multicast group " + group + ":" + mkString(port) );

  MulticastConnection *mc = scinew MulticastConnection(const_cast<char *>(group.c_str()), port );


  multicastConnection = scinew NetConnection( *mc,
					      group + ":" + mkString(port),
					      READ_ONLY );
  
  if ( multicastConnection == NULL )
    Log::log( ERROR, "[NetInterface::connectToMulticast] No multicast connection!" );
  
  Log::log( LEAVE, "[NetInterface::connectToMulticast] leaving, thread id = " + mkString((int) pthread_self()) );
}

bool
NetInterface::connectToServer( string server, int port ) {
  Log::log( ENTER, "[NetInterface::connectToServer] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( MESSAGE, 
	    "[NetInterface::connectToServer] Connecting to server " + server + ":" + mkString(port) );

  // Register for handshake and multicast messages.
  handshakeID = NetDispatchManager::getInstance().
    registerCallback( HANDSHAKE,
		      ClientProperties::setServerProperties,
		      NULL,
		      true );
  multicastID = NetDispatchManager::getInstance().
    registerCallback( MULTICAST,
		      NetInterface::__enableMulticast,
		      this,
		      true );
  // Register callback for data transfer mode message
  
  transferID = NetDispatchManager::getInstance().
    registerCallback( TRANSFER,
		      NetInterface::__transferCallback,
		      this,
		      true );

  PTPConnection *ptpc = scinew PTPConnection( server, port );
  if ( !ptpc->valid() ) {
    Log::log( ERROR, "[NetInterface::connectToServer] Cannot connect to  server!" );
    delete ptpc;
    NetDispatchManager::getInstance().deleteCallback( handshakeID );
    NetDispatchManager::getInstance().deleteCallback( multicastID );
    return false;
  }
    
  serverConnection = scinew NetConnection( *ptpc,
					   server + ":" + mkString(port) );
  
  if ( serverConnection == NULL )
    Log::log( ERROR, "[NetInterface::connectToServer] No server connection!" );
  
  // Also send handshake.
  sendDataToServer( ClientProperties::mkHandshake() ); 

  Log::log( LEAVE, "[NetInterface::connectToServer] leaving, thread id = " + mkString((int) pthread_self()) );
  return true;
}



void
NetInterface::disconnectFromMulticast() {
  Log::log( ENTER, "[NetInterface::disconnectFromMulticast] entered, thread id = " + mkString((int) pthread_self()) );
  // Close the connetion.
  if ( multicastConnection != NULL )
    multicastConnection->close();
  
  // Set the connection to unconnected.
  multicastConnection = NULL;
  
  // LocalUIManager.getInstance().setMulticastGroup( "Unicast" );

  Log::log( MESSAGE, "[NetInterface::disconnectFromMulticast] Disconnected from multicast" );
  Log::log( LEAVE, "[NetInterface::disconnectFromMulticast] leaving, thread id = " + mkString((int) pthread_self()) );
}

void 
NetInterface::disconnectFromServer() {
  Log::log( ENTER, "[NetInterface::disconnectFromServer] entered, thread id = " + mkString((int) pthread_self()) );
  // Close the connection.
  if ( serverConnection != NULL )
    serverConnection->close();
  
  // Set the connection to unconnected.
  serverConnection = NULL;

  // Remove the callbacks
  NetDispatchManager::getInstance().deleteCallback( handshakeID );
  NetDispatchManager::getInstance().deleteCallback( multicastID );
  
  Log::log( MESSAGE, "[NetInterface::disconnectFromServer] Disconnected from server" );
  Log::log( LEAVE, "[NetInterface::disconnectFromServer] leaving, thread id = " + mkString((int) pthread_self()) );
}

void 
NetInterface::enableMulticast( const Multicast &m ) {
  Log::log( ENTER, "[NetInterface::enableMulticast] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, "[NetInterface::enableMulticast] Enable Multicast" );
    
  if ( m.isDisconnect() ) {
    disconnectFromMulticast();
  }
  else {
    string group = m.getGroup();
    int    port  = m.getPort();
    //int    ttl   = m.getTTL();

    cerr << "NetInterface::enableMulticast - group = " << group << ", port = " << port << endl;

    Multicast *response = new Multicast();

    if ( multicastEnabled ) {
      connectToMulticast( group, port );
      response->setConfirm( true );
    }
    else
      response->setConfirm( false );

    response->finish();

    sendDataToServer( response );
    delete response;
  }
   Log::log( LEAVE, "[NetInterface::enableMulticast] leaving, thread id = " + mkString((int) pthread_self()) );
}

void NetInterface::setTransferMode( MessageData * message ) {
  Log::log( ENTER, "[NetInterface::setTransferMode] entered" );
  Transfer *t = (Transfer *)(message->message);

  if ( t->isTransferValid() ){
    // set member variable
    transferMode = t->getTransferType();

    // set UI variable (appears below image on UI)
    UserInterface::getHelper()->setTransfer( transferMode ); 
    char *buffer = scinew char[ transferMode.length() + 100 ];
    sprintf( buffer, "New data transfer mode: %s", const_cast<char*>(transferMode.c_str()) );
    Log::log( DEBUG, buffer );
  }
  else {
    Log::log( ERROR, "[NetInterface::setTransferMode] Invalid data transfer mode sent to server" );
  }
  Log::log( LEAVE, "[NetInterface::setTransferMode] leaving" );
}

char * 
NetInterface::getDataFromMulticast( int &numBytes ) {
  Log::log( ENTER, "[NetInterface::getDataFromMulticast] entered, thread id = " + mkString((int) pthread_self()) );
  unsigned bytesToReadRaw=0, bytesToRead=0;
  multicastConnection->read( (char *)&bytesToReadRaw,
			     sizeof(unsigned int) );
  
  HomogenousConvertNetworkToHost( (void *)bytesToRead,
				  (void *)bytesToReadRaw,
				  UNSIGNED_INT_TYPE,
				  1 );  
  if ( bytesToRead > 4000000 ) 
    Log::log( WARNING, "[NetInterface::getDataFromMulticast] Trying to allocate a LOT of memory: " + 
	     mkString(bytesToRead) + " bytes.");
  char * data = scinew char[ bytesToRead ];
  
  numBytes = multicastConnection->read( data, bytesToRead );

  Log::log( LEAVE, "[NetInterface::getDataFromMulticast] leaving, thread id = " + mkString((int) pthread_self()) );
  
  return data;
}

char * 
NetInterface::getDataFromServer( int &numBytes ) {
  Log::log( ENTER, "[NetInterface::getDataFromServer] entered, thread id = " + mkString((int) pthread_self()) );
  unsigned bytesToReadRaw=0, bytesToRead=0;

  serverConnection->read( (char *)&bytesToReadRaw,
			  sizeof(unsigned int) );

  if ( bytesToRead > 4000000 ) 
    Log::log( WARNING, "[NetInterface::getDataFromServer] Trying to allocate a LOT of memory: " + 
	     mkString(bytesToRead) + " bytes.");
  char * data = scinew char[ bytesToRead ];
  
  numBytes = serverConnection->read( data, bytesToRead );

  Log::log( LEAVE, "[NetInterface::getDataFromServer] leaving, thread id = " + mkString((int) pthread_self()) );
  return data;
}

void 
NetInterface::sendDataToServer( const string &data, const int numBytes ) {
  Log::log( ENTER, "[NetInterface::sendDataToServer] entered, thread id = " + mkString((int) pthread_self()) );
  Log::log( DEBUG, data );
  
  if (serverConnection == NULL)
    Log::log( WARNING, "Not writing to unconnected server connection");
  else {
    serverConnection->write(data.data(), numBytes);
  }
  Log::log( LEAVE, "[NetInterface::sendDataToServer] leaving, thread id = " + mkString((int) pthread_self()) );
}


NetInterface&
NetInterface::getInstance() {
  return net;
}

}
