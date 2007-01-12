/*
 *
 * NetInterface: Provides access to the network
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
#include <Properties/ClientProperties.h>
#include <Rendering/RenderGroup.h>
#include <Rendering/Renderer.h>

using namespace SemotusVisum::Logging; 
using namespace SemotusVisum::Properties;
using namespace SemotusVisum::Rendering;

using namespace std;

namespace SemotusVisum {
namespace Network {

using namespace Message;
using namespace SCIRun;

NetListener::NetListener( NetInterface &net, int port ) :
  dieNow( false ), port(port), net(net), mbox(net.connectMBox) {
  cerr << "In NetListener::NetListener, thread id is " << pthread_self() << endl;
  /* Create a socket */
  listenSocket = OpenTcpPort( ANY_ADDRESS, port );
  if ( listenSocket == ASIP_NO_SOCKET )
    Log::log( Logging::ERROR, "Cannot open listening socket!" );
  cerr << "End of NetListener::NetListener, thread id is " << pthread_self() << endl;
}

NetListener::~NetListener() {
  cerr << "In NetListener destructor, thread id is " << pthread_self() << endl;
  /* Close our listening socket */
  Disconnect( &listenSocket );
  cerr << "End of NetListener destructor, thread id is " << pthread_self() << endl;
}

void
NetListener::run() {
  cerr << "In NetListener::run, thread id is " << pthread_self() << endl;
  ASIP_Socket newSocket = NO_SOCKET;
  PTPConnection *ptpc = NULL;
  char * clientName = NULL;

  std::cerr << "NetListener: " << getpid() << endl;
  //Thread::self()->makeInterruptable();
  
  for (;;) {
    
    /* Listen to the socket */
    newSocket = Accept( listenSocket );

    /* If we need to exit, do that now. */
    if ( dieNow == true ) {
      Disconnect( &newSocket );
      return;
    }
    
    Log::log( Logging::MESSAGE, "Got a client connection." );
    
    if ( newSocket != NO_SOCKET ) {
      /* When we get a connection, notify the NetInterface */
      
      ptpc = scinew PTPConnection( newSocket );
      
      if ( ptpc != NULL ) {
	clientName =
	  const_cast<char *>( AddressImage( PeerAddress( newSocket ) ) );
	if ( clientName[0] != 0 )
	  net.addConnection( *ptpc, clientName );
	else
	  Log::log( Logging::ERROR,
		    "Could not get client host name!" );
      }
      else {
	Log::log( Logging::ERROR,
		  "Could not create new point-to-point connection!" );
      }
    }
    else {
      /* Error - log it */
      Log::log( Logging::ERROR, "Could not accept socket connection!");
    }
    
  }
  cerr << "End of NetListener::run, thread id is " << pthread_self() << endl;
}

  /* NetInterface Code */

/* The singleton Network Interface */
NetInterface *
NetInterface::net = NULL;

/* Data marker for the end of XML data and the beginning of raw data. */
const char * const
NetInterface::dataMarker = "\001\002\003\004\005";

NetInterface::NetInterface() :
  connectMBox( "ConnectionMailbox", MAX_PENDING_ITEMS ),
  netConnectionLock( "NetConnectionMutex" ),
  haveConnections( "ConnectionBlock" ),
  remoteModuleID( -1 ),
  enableMulticast( true ),
  listenThread( NULL ) {
  cerr << "In NetInterface::NetInterface, thread id is " << pthread_self() << endl;
  cerr << "Starting Net Interface" << endl;
  haveConnections.lock(); // We have no connections yet.
  
  /* Add a callback to get the handshake info from clients. */
  NetDispatchManager::getInstance(). 
    registerCallback( Message::HANDSHAKE,
		      NetInterface::getHandshake,
		      NULL,
		      true );

  transferMode = "PTP";
  cerr << "End of NetInterface::NetInterface, thread id is " << pthread_self() << endl;
}
  
NetInterface::~NetInterface() {
  cerr << "In NetInterface destructor, thread id is " << pthread_self() << endl;
  /* Unlock the HaveConnections mutex... */
  haveConnections.unlock();

  /* Note - we don't need to do this. As the net interface is static, this
     is only called upon shutdown... */
  return;
#if 0  
  /* Take care of the network listener */
  if ( listenThread ) {
    listener->dieNow = true;
    /* FIXME - we need a way to kill the listening thread. */
    //listenThread->interrupt(); 
    listenThread->join();
  }
  delete listener;
#endif
  cerr << "End of NetInterface destructor, thread id is " << pthread_self() << endl;
}


void
NetInterface::listen(int port) {
  cerr << "In NetInterface::listen, thread id is " << pthread_self() << endl;
  /* Create a network listener */
  listener = scinew NetListener( *this, port );

  /* Bind it to a thread */
  listenThread = scinew Thread( listener, "NetListener" );
  cerr << "End of NetInterface::listen, thread id is " << pthread_self() << endl;
}


void
NetInterface::stop() {
  cerr << "In NetInterface::stop, thread id is " << pthread_self() << endl;
  if ( !listener )  return; // No connections!

  /* First, we stop listening for new connections. */
  Log::log( Logging::DEBUG, "Stopping listen thread" );
  listener->dieNow = true;
    /* FIXME - we need a way to kill the listening thread. */
  unsigned int address;
  AddressValues( "localhost", &address, 1 );
  ConnectToTcpPort( address, listener->getPort() );
  
  listenThread->join();
  Log::log( Logging::DEBUG, "Listen thread stopped." );
  
  //delete listener;
  listener = NULL;
  listenThread = NULL;
  
  /* Then, we finish up anything we have to do with our existing connections,
     and close them. */
  NetConnection::getNetMonitor().getMailbox().send( NULL );
  cerr << "End of NetInterface::stop, thread id is " << pthread_self() << endl;
}


void   
NetInterface::listenMulticast(int port) {
  cerr << "In NetInterface::listenMulticast, thread id is " << pthread_self() << endl;
  // Right now, we don't listen to the multicast group - we just write...
  cerr << "End of NetInterface::listenMulticast, thread id is " << pthread_self() << endl;
}

void
NetInterface::onConnect( void * obj, connectFunc func ) {
  cerr << "In NetInterface::onConnect, thread id is " << pthread_self() << endl;
  connectFunctions.push_back( func );
  connectData.push_back( obj );
  cerr << "End of NetInterface::onConnect, thread id is " << pthread_self() << endl;
}

void
NetInterface::enableRemoteModuleCallback( bool enable ) {
  cerr << "In NetInterface::enableRemoteModuleCallback, thread id is " << pthread_self() << endl;
  if ( enable && remoteModuleID != -1 ) {
    
    // Remove our default callback.
    NetDispatchManager::getInstance().deleteCallback( remoteModuleID );
    remoteModuleID = -1;
    
  }
  else if ( !enable && remoteModuleID == -1 ) {

    // Add our default callback.
    remoteModuleID =
      NetDispatchManager::getInstance().registerCallback( XDISPLAY,
					    NetInterface::remoteModuleCallback,
					    NULL,
					    true );
  }
  cerr << "End of NetInterface::enableRemoteModuleCallback, thread id is " << pthread_self() << endl;
}

void
NetInterface::remoteModuleCallback( void * obj, MessageData *input ) {
  cerr << "In NetInterface::remoteModuleCallback, thread id is " << pthread_self() << endl;
  // Create an XDisplay message
  XDisplay x;

  // We don't even need to look at our incoming message. The answer is
  // always no. :)
  x.setResponse( false, "Function not implemented." );
  x.finish();

  // Send the reply.
  if ( net )
    net->sendPriorityDataToClient( input->clientName, x );
  cerr << "End of NetInterface::remoteModuleCallback, thread id is " << pthread_self() << endl;
}


void
NetInterface::getHandshake( void * obj, MessageData *input ) {
  std::cerr << "In NetInterface::getHandshake, thread id is " << pthread_self() << endl;
  std::cerr << " =========GOT HANDSHAKE FROM CLIENT!============ " << endl;
  Log::log( Logging::DEBUG, "Receiving client handshake data.");

  // Create a handshake from the data.
  Handshake * h = (Handshake *)(input->message);
  
  if ( h == NULL ) {
    Log::log( Logging::ERROR, "Problem with handshake" );
  }
  else {
    // Create a client properties object
    ClientProperties * cp = scinew ClientProperties;
    
    // Populate it
    if ( ClientProperties::getFormats( *h, *cp ) == false ) {
      
      Log::log( Logging::ERROR, "Problem with handshake data" );
      
      // There was a problem with the handshake. Delete the object.
      delete cp;
      delete h;
    }
    else {
      // Grab the nickname of the client.
      char * nick = h->getNickname();
      if ( nick == NULL )
	nick = "NoName";

#if 1
      /* Send a message to all clients about the addition. */
      GetClientList g;
      g.clientAdded( nick, input->clientName );
      g.finish();
      char buffer[1000];
      snprintf( buffer, 1000, "Sending client add of %s:%s to all clients\n",
		nick, input->clientName );
      Log::log( Logging::DEBUG, buffer );
      if ( net ) 
	net->sendDataToAllClients( g );
#endif
      // Now set the nickname of the net connection.
      NetConnection *nc=NULL;
      if ( net )
	nc = net->getConnection( input->clientName );
      if ( nc == NULL ) {
	Log::log( Logging::ERROR,
		  "No net connection with the given client name in handshake");
      }
      else {
	char buffer[1000];
	snprintf(buffer, 1000, "Client %s's nickname is %s",
		 input->clientName, nick );
	Log::log( Logging::MESSAGE, buffer );
	nc->setNickname( nick );
      }
    }
    
    // We automatically add the client properties object to a static
    // list at creation. Thus, we are done.
  }
  cerr << "End of NetInterface::getHandshake, thread id is " << pthread_self() << endl;
}

void
NetInterface::modifyClientGroup( const char * clientName,
				 const char * group ) {
  cerr << "In NetInterface::modifyClientGroup, thread id is " << pthread_self() << endl;
  
  char * nick = NULL;
  
  // Grab the client's nickname.
  NetConnection *nc = NULL;
  if ( net )
    nc = net->getConnection( clientName );
  if ( nc == NULL ) {
    Log::log( Logging::ERROR,
	      "No net connection with the given client name in modclientgrp");
    return;
  }
  else 
    nick = nc->getNickname();

  GetClientList g;
  g.clientModified( nick, clientName, group );
  g.finish();
  
  // Push out the modification
  if ( net )
    net->sendPriorityDataToAllClients( g );
  cerr << "End of NetInterface::modifyClientGroup, thread id is " << pthread_self() << endl;
}


void
NetInterface::getClientList( void * obj, MessageData *input ) {
  cerr << "In NetInterface::getClientList, thread id is " << pthread_self() << endl;
  if ( !net ) return;
  // We assume that the client list message is okay.

  // Build a client list.
  GetClientList g;
  net->netConnectionLock.readLock();

  RenderGroup *rg = NULL;
  for ( list<NetConnection *>::iterator i = net->clientConnections.begin();
	i != net->clientConnections.end(); i++ ) {
    if ( rg = RenderGroup::getRenderGroup( input->clientName ) )
      if ( rg->getName() != NULL ) {
	g.addClient( (*i)->getNickname(), (*i)->getName(), rg->getName() );
	continue;
      }
    g.addClient( (*i)->getNickname(), (*i)->getName() );
  }
  
  g.finish();
  
  net->netConnectionLock.readUnlock();

  // Send the list to the new client
  net->sendPriorityDataToClient( input->clientName, g );
  cerr << "End of NetInterface::getClientList, thread id is " << pthread_self() << endl;
}

void
NetInterface::transferCallback( void * obj, MessageData *input ) {
  cerr << "In NetInterface::transferCallback" << endl;
  Log::log( Logging::DEBUG, "Got a transfer mode message!" );

  if ( !net ) return;

  // Create a transfer message from the data.
  Transfer * t = (Transfer *)(input->message);
  
  if ( t == NULL ) {
    Log::log( Logging::ERROR, "Problem with transfer message" );
  }
  else {    
    // update the transfer mode based on type sent
    net->transferMode = t->getName();

    // check to make sure this is a valid transfer mode
    if((net->transferMode != "PTP") && (net->transferMode != "RM") && (net->transferMode != "IPM")){
      Log::log( Logging::DEBUG, "Unknown transfer mode, switching to default (PTP)" );
      net->transferMode = "PTP";
    }
    char *buffer = scinew char[ net->transferMode.length() + 100 ];
    sprintf( buffer, "New data transfer mode: %s", const_cast<char*>(net->transferMode.c_str()) );
    Log::log( Logging::DEBUG, buffer );

    // send updated transfer mode to all clients
    // FIXME -- This causes a seg fault for some reason
    //net->sendDataToAllClients("TEST DATA", CHAR_TYPE, 9, true);
    //char * test = scinew char[200];
    //test = "<?xml version='1.0' encoding='ISO-8859-1' ?><transfer>PTP</transfer>";
    //net->sendDataToAllClients(test, CHAR_TYPE, strlen(test), true);

    // Send new transfer mode to all clients
    Transfer *reply = scinew Transfer(false);
     
    bool okay = true; 
    reply->setOkay( okay, t->getName() );
    reply->finish();

    net->sendDataToAllClients(*reply);

  }  

  cerr << "End of NetInterface::transferCallback" << endl;  
}

void
NetInterface::getCollaborate( void * obj, MessageData *input ) {
  cerr << "In NetInterface::getCollaborate, thread id is " << pthread_self() << endl;
  Log::log( Logging::DEBUG, "Got a collaborate message!" );

  if ( !net ) return;
  
  // Create a collaborate message from the data.
  Collaborate * c = (Collaborate *)(input->message);

  if ( c == NULL ) {
    Log::log( Logging::ERROR, "Problem with collaborate" );
  }
  else {
    // Look up client that sent the message
    char * client = input->clientName;
    
    // Get that client's render group.
    RenderGroup *rg = RenderGroup::getRenderGroup( client );

    // If the client isn't in a rendering group, punt.
    if ( rg == NULL ) {
      char buffer[256];
      snprintf( buffer, 256, "Client %s is not in a rendering group!",
		client );
      Log::log( Logging::WARNING, buffer );
      return;
    }
    
    // Build a new outgoing collaborate message
    Collaborate Cout(true);

    Cout.setText( c->getText() );
    char * tmpBuf = scinew char[ strlen(client) + 6 ];
    sprintf( tmpBuf, "%sRemote", client );
    Cout.switchID( tmpBuf );
    delete tmpBuf;
    Cout.finish();
    
    // Relay chat info to the clients in the render group.
    if ( rg->getMulticastGroup() != NULL ) {
      if ( !net->sendDataToGroup( rg->getMulticastGroup(), Cout ) )
	Log::log( Logging::ERROR,
		  "Problem sending collaborate message to multicast group!" );
    }
    else {
      // Send it back to everyone except the current client.
      
      list<NetConnection*>::iterator i;
      list<char *>::const_iterator j;
      list<char *>& clients = rg->getClients();
      for ( i = net->clientConnections.begin();
	    i != net->clientConnections.end();
	    i++ ) {
	for ( j = clients.begin(); j != clients.end(); j++ )
	  if ( !strcmp( (*i)->getName(), (char *)*j ) &&
	       strcmp((char *)*j, client ) )
	    if ( !net->sendPriorityDataToClient( (*i)->getName(), Cout ) )
	      break;
	
      }
    }
  }
  cerr << "End of NetInterface::getCollaborate, thread id is " << pthread_self() << endl;
}

void
NetInterface::getChat( void * obj, MessageData *input ) {
  cerr << "In NetInterface::getChat, thread id is " << pthread_self() << endl;
  Log::log( Logging::DEBUG, "Got a chat message!" );
  
  // Create a chat message from the data.
  Chat * c = (Chat *)(input->message);

  if ( c == NULL ) {
    Log::log( Logging::ERROR, "Problem with chat" );
  }
  else {
    // Look up client that sent the message
    char * client = input->clientName;
    NetConnection *nc = net->getConnection( client );
    char * clientName = NULL;
    
    if ( nc == NULL ) {
      Log::log( Logging::ERROR, "Got a chat message from an unknown client" );
      clientName = strdup( client );
    }
    else {
      char * nick = nc->getNickname();
      if ( nick == NULL ) {
	clientName = strdup( client );
      }
      else {
	clientName = scinew char[ strlen( client ) + strlen( nick ) + 10 ];
	sprintf(clientName, "%s@%s", nick, client );
      }
    }
    
    // Build a new outgoing chat message
    Chat Cout(true);

    Cout.setName( clientName );
    Cout.setText( c->getText() );
    Cout.finish();
    
    // Relay chat info to clients
    net->sendPriorityDataToAllClients( Cout );
    
    delete clientName;
  }
  cerr << "End of NetInterface::getChat, thread id is " << pthread_self() << endl;
}

bool
NetInterface::sendDataToClient(const char * clientName,
			       const char * data,
			       const DataTypes dataType,
			       const int numBytes,
			       bool copy )  {
  cerr << "In NetInterface::sendDataToClient, thread id is " << pthread_self() << endl;
  return realSendDataToClient( clientName,
			       data,
			       dataType,
			       numBytes,
			       copy,
			       false ); // Not a priority message
  cerr << "End of NetInterface::sendDataToClient, thread id is " << pthread_self() << endl;
}


bool
NetInterface::sendPriorityDataToClient(const char * clientName,
				       const char * data,
				       const DataTypes dataType,
				       const int numBytes,
				       bool copy )  {
  cerr << "In NetInterface::sendPriorityDataToClient, thread id is " << pthread_self() << endl;
  cerr << "End of NetInterface::sendPriorityDataToClient, thread id is " << pthread_self() << endl;
  return realSendDataToClient( clientName,
			       data,
			       dataType,
			       numBytes,
			       copy,
			       true ); // A priority message
 
  
}

bool
NetInterface::realSendDataToClient(const char * clientName, 
				   const char * data,
				   const DataTypes dataType,
				   const int numBytes,
				   bool copy,
				   bool priority)  {
  cerr << "In NetInterface::realSendDataToClient, thread id is " << pthread_self() << endl;
  
  /* Find the client in the list of network connections */
  list<NetConnection*>::iterator i;
  NetConnection *connection = NULL;
  
  for ( i = clientConnections.begin();
        i != clientConnections.end();
        i++ ) {

    if ( !( strcmp( clientName, (*i)->getName() ) ) ) {
      connection = *i;
      break;
    }
    
  }
 
 
  char buffer[1000];
  snprintf(buffer, 1000, "Trying to send data to %s", clientName );
  Log::log( Logging::DEBUG, buffer );
  /* If the client exists, write the data to the connection */
  if ( connection != NULL ) {

    if ( copy )
      Log::log( Logging::DEBUG, "Sending copied data!");
    else
      Log::log( Logging::DEBUG, "Sending original data!" );
    
    bool result;
    //    cerr << "\tsending data" << endl;

    /* Now, we check if there's room left in the mailbox. We keep the
       last PRIORITY_BIN_SIZE slots free... */
    // This ought to be protected by a mutex, but what the hell...
    if ( !priority &&
	 MAX_PENDING_ITEMS -
	 connection->getMailbox().numItems() <= PRIORITY_BIN_SIZE )
      result = false;
    else if ( priority && connection->getMailbox().numItems() ==
	      MAX_PENDING_ITEMS ){
      // If the mailbox is full and this is a priority message,
      // then add it to the priority list.
      if ( copy ) {
	char * _data = scinew char[ numBytes ];
	memcpy( _data, data, numBytes );
	connection->getPriorityList().push_back( dataItem(_data,
							numBytes,
							true,
							dataType ) );
      }
      else {
	connection->getPriorityList().push_back( dataItem(data,
							  numBytes,
							  false,
							  dataType ) );
      }
    }
    else  {
      
      if ( copy ) {
	char * _data = scinew char[ numBytes ];
	memcpy( _data, data, numBytes );
	result =
	  connection->getMailbox().trySend( dataItem( _data,
						      numBytes,
						      true,
						      dataType ) );
      }
      else 
	result =
	  connection->getMailbox().trySend( dataItem( data,
						      numBytes,
						      false,
						      dataType ) );
    }
    
    /* If the mailbox was full, log it. */
    if ( result == false ) {
      char *buffer = scinew char[ strlen(clientName) + 256 ];
      sprintf( buffer,
	       "Dropped data send to client: %s - Mailbox full [%d/%d]",
	       clientName,
	       connection->getMailbox().numItems(),
	       connection->getMailbox().size() );
      
      Log::log( Logging::WARNING, buffer );
      delete buffer;
    }
    cerr << "End of NetInterface::realSendDataToClient, thread id is " << pthread_self() << endl;
    return true;
  }
  
  /* Otherwise log the error */
  else {
    char *buffer = scinew char[ strlen(clientName) + 40 ];
    sprintf( buffer, "Data sent to unknown client: %s", clientName );
    
    Log::log( Logging::WARNING, buffer );
    delete buffer;

    cerr << "End of NetInterface::realSendDataToClient, thread id is " << pthread_self() << endl;
    return false;
  }
  cerr << "End of NetInterface::realSendDataToClient, thread id is " << pthread_self() << endl;
}

bool
NetInterface::sendDataToClients( list<char *>  &clients,
				 const char * data,
				 const DataTypes dataType,
				 const int numBytes,
				 bool  copy ) {
  cerr << "In NetInterface::sendDataToClients, thread id is " << pthread_self() << endl;
  return realSendDataToClients( clients,
				data,
				dataType,
				numBytes,
				copy,
				false ); // Not a priority message
  cerr << "End of NetInterface::sendDataToClients, thread id is " << pthread_self() << endl;
}

bool
NetInterface::sendPriorityDataToClients( list<char *>  &clients,
					 const char * data,
					 const DataTypes dataType,
					 const int numBytes,
					 bool  copy ) {
  cerr << "In NetInterface::sendPriorityDataToClients, thread id is " << pthread_self() << endl;
  return realSendDataToClients( clients,
				data,
				dataType,
				numBytes,
				copy,
				true ); // A priority message
   cerr << "End of NetInterface::sendPriorityDataToClients, thread id is " << pthread_self() << endl;
}

bool
NetInterface::realSendDataToClients( list<char *>  &clients,
				     const char * data,
				     const DataTypes dataType,
				     const int numBytes,
				     bool  copy,
				     bool  priority ) {
  
  cerr << "In NetInterface::realSendDataToClients, thread id is " << pthread_self() << endl;
  // Send data via PTP.
  list<NetConnection*>::iterator i;
  list<char *>::const_iterator j;
  bool retval = true; // OK
  
  for ( i = clientConnections.begin();
	i != clientConnections.end();
	i++ ) {
    for ( j = clients.begin(); j != clients.end(); j++ )
      if ( !strcmp( (*i)->getName(), (char *)*j ) ) {
	if ( !realSendDataToClient( (*i)->getName(), data,
				    dataType, numBytes, copy, priority ) )
	  retval = false;
	break;
      }
  }

  cerr << "End of NetInterface::realSendDataToClients, thread id is " << pthread_self() << endl; 
  return retval;
}

multicastGroup * 
NetInterface::createMulticastGroup( list<char *> &clients,
				    const char * _group, const int _port ) {
  cerr << "In NetInterface::createMulticastGroup, thread id is " << pthread_self() << endl;
  static bool callbackRegistered = false;
  char * group;
  int port;
  int ttl;
  char buffer[ 100 ];

  std::cerr << "Port: " << _port << endl;
  
  // Find a clear group:port, ttl unless they're specified.
  if ( _group == NULL || _port == -1 )
    getMulticastGroup( group, port, ttl );
  else {
    group = strdup(_group); port = _port;
  }

  // Create a multicast connection
  MulticastConnection *mc = scinew MulticastConnection( group, port );

  std::cerr << "Created MC: " << group << ":" << port << endl;
  snprintf( buffer, 100, "Created multicast group %s:%d", group, port );
  Log::log( Logging::MESSAGE, buffer );
  
  // Create a net connection
  snprintf( buffer, 100, "%s:%d", group, port );
  NetConnection * nc = scinew NetConnection( *mc, buffer, WRITE_ONLY );
  
  // Create the multicast group
  multicastGroup *mg = scinew multicastGroup( nc,
					   group,
					   port );
  multicastClients.push_front( mg );

  
  std::cerr << "Created NC" << endl;
  // Set up callback from dispatch manager.
  if ( !callbackRegistered ) {
    

    NetDispatchManager::getInstance().
    registerCallback( MULTICAST,
       			multicastResponseCallback,
      			(void *)net,
      			true );

    
    callbackRegistered = true;
  }
  std::cerr << "Created Callback" << endl;
  // Add clients to multicast group
  list<char *>::const_iterator i;
  for ( i = clients.begin(); i != clients.end(); i++ )
    addToMulticastGroup( (char *)*i, mg );

  cerr << "End of NetInterface::createMulticastGroup, thread id is " << pthread_self() << endl;
  return mg;
}

 
void 
NetInterface::addToMulticastGroup( const char * client, multicastGroup * mg ) {
  cerr << "In NetInterface::addToMulticastGroup, thread id is " << pthread_self() << endl;
  // Send enable message to client
  Multicast m( false );
  m.setParams( mg->name, mg->port, 5 );
  m.finish();
  std::cerr << "Created MM" << endl;
  sendDataToClient( client, m );
  std::cerr << "Sent MM" << endl;
  // Place clients in a 'multicast possible' list
  multicastPossible * mp =
    scinew multicastPossible( client, mg->name, mg->port );
  std::cerr << "Created MP" << endl;
  //multiPossibles.push_front( *mp );
  mg->addOtherClient( mp );
  cerr << "End of NetInterface::addToMulticastGroup, thread id is " << pthread_self() << endl;
}

void
NetInterface::deleteFromMulticastGroup( const char * client,
					multicastGroup *mg ) {
  cerr << "In NetInterface::deleteFromMulticastGroup, thread id is " << pthread_self() << endl;
  if ( mg->deleteClient( client ) ) {
    
    // Send a disable message to client.
    Multicast m( false );
    m.setDisconnect( true );
    m.finish();
    
    sendDataToClient( client, m );
  }
  else 
    Log::log( Logging::WARNING,
	      "Trying to remove a client that is not in a multicast group" );
  cerr << "End of NetInterface::deleteFromMulticastGroup, thread id is " << pthread_self() << endl;
}
 
  
bool
NetInterface::sendDataToGroup( const multicastGroup *mg, const char * data,
			       const DataTypes dataType, const int numBytes ) {
  
  cerr << "In NetInterface::sendDataToGroup, thread id is " << pthread_self() << endl;
  bool result = true;

  if ( !mg ) {
    Log::log( Logging::WARNING, "No multicast group in sendDataToGroup()" );
    return false;
  }


  if(SV_TRANSFER_MODE == 0){
    //
    // PTP code
    //

  
    //if ( numBytes > MAXIMUM_DATA_SIZE ) {
      char buffer[ 256 ];
      snprintf( buffer, 256,
	      "Multicast message of size %d is too large! Reverting to PTP",
	      numBytes );
      Log::log( Logging::WARNING, buffer );
  
      cerr << "Reverting to PTP" << endl;
      list<char *>::const_iterator i;
      bool retval = true;
      for ( i = mg->clientNames.begin(); i != mg->clientNames.end(); i++ ){
        cerr << "Sending " << numBytes << " bytes of data via PTP" << endl;
        if ( sendDataToClient( *i, data, dataType, numBytes ) == false ){ 
       	  retval = false;
        }
      }
      cerr << "End of NetInterface::sendDataToGroup, thread id is " << pthread_self() << endl;
      return retval;
      //}
  }
  else if(SV_TRANSFER_MODE == 1 || SV_TRANSFER_MODE == 2){

    //
    // Multicast code
    //

        
    // First, send data to everybody in the multicast group 
  
    cerr << "Attempting to send data to multicast group with address: " << mg->name << " port: " << mg->port << endl;
    cerr << "Data: " << data << endl;
   result = mg->group->getMailbox().trySend( dataItem( data, numBytes, false, dataType ) );
   
  
    // If the mailbox was full, log it. 
  
    if ( result == false ) {
      char *buffer = scinew char[ strlen(mg->name) + 256 ];
      snprintf( buffer, strlen(mg->name) + 256,
	      "Dropped data send to multicast: %s:%d - Mailbox full [%d/%d]",
	      mg->name, mg->port,
	      mg->group->getMailbox().numItems(),
	      mg->group->getMailbox().size() );
    
      Log::log( Logging::WARNING, buffer );
      delete buffer;
    }
         
   

    /*
    // Now, send data to everybody in the PTP group
    // BROKEN CODE
 
    cerr << "Sending to multicastPosslibe clients via PTP" << endl;
    list<multicastPossible*>::const_iterator j;

    bool retval = true;
    char buffer[256];
    for ( j = mg->otherClients.begin(); j != mg->otherClients.end(); j++ )
      cerr << "Sending PTP to client in multicastPossible list" << endl;
      if ( sendDataToClient( (*j)->clientName, data, dataType, numBytes ) ==
	 false ) {
        snprintf( buffer, 256, "Cannot send PTP data to %s in sendDataToGroup()",
		(*j)->clientName );
        Log::log( Logging::ERROR, buffer );
        retval = false;
      }
   
    cerr << "End of NetInterface::sendDataToGroup, thread id is " << pthread_self() << endl;
  

    return retval;
    */ 
  }
  else{
    Log::log( Logging::ERROR, "Unrecognized data transfer mode" );
  }

  return true;
}

void
NetInterface::sendDataToAllClients(const char * data,
				   const DataTypes dataType,
				   const int numBytes,
				   bool copy ) {
  cerr << "In NetInterface::sendDataToAllClients, thread id is " << pthread_self() << endl;
  
  realSendDataToAllClients( data,
			    dataType,
			    numBytes,
			    copy,
			    false ); // Not a priority message
  cerr << "End of NetInterface::sendDataToAllClients, thread id is " << pthread_self() << endl;
  
}

void
NetInterface::sendPriorityDataToAllClients(const char * data,
					   const DataTypes dataType,
					   const int numBytes,
					   bool copy ) {
  cerr << "In NetInterface::sendPriorityDataToAllClients, thread id is " << pthread_self() << endl;

  realSendDataToAllClients( data,
			    dataType,
			    numBytes,
			    copy,
			    true ); // A priority message
  cerr << "End of NetInterface::sendPriorityDataToAllClients, thread id is " << pthread_self() << endl;
}

void
NetInterface::realSendDataToAllClients(const char * data,
				       const DataTypes dataType,
				       const int numBytes,
				       bool copy,
				       bool priority ) {
  cerr << "In NetInterface::realSendDataToAllClients, thread id is " << pthread_self() << endl;
  cerr << "Number of multicast groups = " << multicastClients.size() << endl;
  char buffer[1000];
  Log::log( Logging::DEBUG, "Sending data to all clients!" );
  
  /* Find the client in the list of network connections */
  list<NetConnection*>::iterator i;

  for ( i = clientConnections.begin();
	i != clientConnections.end();
	i++ ) {
    snprintf( buffer, 1000, "Sending data to %s", (*i)->getName() );
    Log::log( Logging::DEBUG, buffer );
    realSendDataToClient( (*i)->getName(), data,  dataType, numBytes, copy,
			  priority );
  }
  cerr << "End of NetInterface::realSendDataToAllClients, thread id is " << pthread_self() << endl;
}



NetInterface&
NetInterface::getInstance() {
  cerr << "In NetInterface::getInstance, thread id is " << pthread_self() << endl;
  
  if ( net == NULL )
    net = scinew NetInterface();
  
  static bool callbackRegistered = false;
  
  // Add callbacks for the goodbye, chat, and getClientList messages.
  if ( !callbackRegistered ) {
    NetDispatchManager::getInstance().
      registerCallback( Message::GOODBYE,
			NetConnection::goodbyeCallback,
			NULL,
			true );
    NetDispatchManager::getInstance().
      registerCallback( Message::CHAT,
			NetInterface::getChat,
			NULL,
			true );
    NetDispatchManager::getInstance().
      registerCallback( Message::GET_CLIENT_LIST,
			NetInterface::getClientList,
			NULL,
			true );
    NetDispatchManager::getInstance().
      registerCallback( Message::COLLABORATE,
			NetInterface::getCollaborate,
			NULL,
			true );
    NetDispatchManager::getInstance().
      registerCallback( Message::TRANSFER,
			NetInterface::transferCallback,
			NULL,
			true );
    
    callbackRegistered = true;			
  }
  cerr << "End of NetInterface::getInstance, thread id is " << pthread_self() << endl;
  return *net;
}

char *
NetInterface::waitForConnections() {
  cerr << "In NetInterface::waitForConnections, thread id is " << pthread_self() << endl;
  haveConnections.lock(); // This blocks until we have a connection.
  haveConnections.unlock(); // Release our hold on this.

  char * temp;
  
  netConnectionLock.readLock();
  temp = (*(clientConnections.begin()))->getName();
  netConnectionLock.readUnlock();

  newConnects = false; // No new connections.
  cerr << "End of NetInterface::waitForConnections, thread id is " << pthread_self() << endl;
  
  return temp;
}

list<char *>*
NetInterface::getClientNames() {
  cerr << "In NetInterface::getClientNames, thread id is " << pthread_self() << endl;
  list<char *> *theList = scinew list<char *>;

  if ( !theList )
    return NULL;

  
  netConnectionLock.readLock();

  // If we have no connections, return NULL.
  if ( clientConnections.size() == 0 ) {
    netConnectionLock.readUnlock();
    delete theList;
    return NULL;
  }

  char * temp;
  
  // For each connection in the list
  for ( list<NetConnection *>::const_iterator i = clientConnections.begin();
	i != clientConnections.end();
	i++ ) {
    
    // Duplicate the name
    temp = strdup( (*i)->getName() );
    
    // Add it to the name list.
    theList->push_front( temp );
  }
  
  
  netConnectionLock.readUnlock();

  cerr << "End of NetInterface::getClientNames, thread id is " << pthread_self() << endl;
  return theList;
}

void
NetInterface::addConnection( Connection &c, const char * hostname ) {
  cerr << "In NetInterface::addConnection, thread id is " << pthread_self() << endl;
  /* Assures all connection names are unique */
  static int connectionNumber = 0;

  char *buffer = scinew char[ strlen( hostname ) + 10 ];
  if ( buffer == NULL ) {
    perror( "Couldn't allocate buffer space!" );
    return;
  }
  sprintf( buffer, "%s:%d", hostname, connectionNumber );
  connectionNumber++;

  // Log the connection
  char *buf = scinew char[ strlen(buffer) + 40 ];
  if ( buf == NULL ) {
    perror( "Couldn't allocate buf space!" );
    return;
  }
  sprintf( buf, "Adding connection to client %s", buffer );
  
  Log::log( Logging::MESSAGE, buf );
  delete buf;

  /* Send client the list of currently connected clients - THIS DOES NOT
     INCLUDE THE CURRENT CLIENT! That is sent later... */
#if 1
  GetClientList g;
  netConnectionLock.readLock();
  
  for ( list<NetConnection *>::iterator i = clientConnections.begin();
	i != clientConnections.end(); i++ ) {
    g.addClient( (*i)->getNickname(), (*i)->getName() );
  }

  g.finish();
  
  netConnectionLock.readUnlock();
#endif
  /* Lock the list of client connections */
  netConnectionLock.writeLock();

  /* Add the client connection */
  clientConnections.push_front( scinew NetConnection( c, buffer ) );
  
  /* Unlock the client list */
  netConnectionLock.writeUnlock();

  /* Send the handshake to the client. */
  ServerProperties::sendHandshake( buffer );
  
  if ( clientConnections.size() == 1 ) // Our first connection.
    haveConnections.unlock();
#if 1
  else 
    // Send list of other clients to new client
    sendDataToClient( buffer, g ); 
#endif

  /* Now we call any onConnect functions that the user has specified */
  std::cerr << "We have " << connectFunctions.size() <<
    " onConnect callbacks. " << endl;
  for ( int i = 0; i < connectFunctions.size(); i++ ) {
    connectFunctions[i]( connectData[ i ], buffer );
  }
  
  delete buffer;
  
  newConnects = true; // We have new connections!
  cerr << "End of NetInterface::addConnection, thread id is " << pthread_self() << endl;
}

void
NetInterface::removeAllConnections() {
  cerr << "In NetInterface::removeAllConnections, thread id is " << pthread_self() << endl;
  Log::log( Logging::MESSAGE, "Removing all connections");
  
  // First remove and delete PTP connections.
  while ( !clientConnections.empty() ) {
    NetConnection * nc = clientConnections.front();

    removeConnection( nc );
    delete nc;
  }
  cerr << "End of NetInterface::removeAllConnections, thread id is " << pthread_self() << endl;
}


void
NetInterface::removeConnection( NetConnection *nc ) {
  cerr << "In NetInterface::removeConnection, thread id is " << pthread_self() << endl;

  std::cerr << "NI: Removing connection " << nc->getName() << endl;
  std::cerr << "Before remove: " << clientConnections.size();

  /* We also remove the connection from any render groups. This is ugly
     and shouldn't be here for cleanliness, but we want to keep this fully
     automated. */
  Rendering::RenderGroup * rg =
    Rendering::RenderGroup::getRenderGroup( nc->getName() );
  if ( rg != NULL ) {
    char buffer[ 1000 ];
    snprintf( buffer, 1000, "Removing client %s from render group %s",
	      nc->getName(),
	      (rg->getRenderer()!=NULL) ?
	           rg->getRenderer()->getName() : "unknown" );
    Log::log( Logging::MESSAGE, buffer );
    rg->removeClient( nc->getName() );
  }
  
  // Remove it from the list - does not delete the connection.
  netConnectionLock.writeLock();
  clientConnections.remove( nc );
  netConnectionLock.writeUnlock();
  Log::log( Logging::DEBUG, "Removing connection from net interface" );
  std::cerr << "\tAfter remove: " << clientConnections.size() << endl;

  
  
  /* FIXME - We also need to remove the client properties for this client
     from the global list. */
  
  
  
  if ( clientConnections.empty() )
    haveConnections.lock(); // We have no connections.
  else {
    /* Send a message to all clients about the subtraction. */
    GetClientList g;
    g.clientSubtracted( nc->getNickname(), nc->getName() );
    g.finish();
    sendDataToAllClients( g );
  }
  cerr << "End of NetInterface::removeConnection, thread id is " << pthread_self() << endl;
}

NetConnection *
NetInterface::getConnection( const char * client ) {
  cerr << "In NetInterface::getConnection, thread id is " << pthread_self() << endl;
  
  if ( client == NULL ) return NULL;
  
  list<NetConnection*>::iterator i;
  NetConnection *retval = NULL;
  
  netConnectionLock.readLock();
  for ( i = clientConnections.begin(); i != clientConnections.end(); i++ ) {
    if ( !strcmp( client, (*i)->getName() ) ) {
      retval = *i;
      break;
    }
  }
  netConnectionLock.readUnlock();
  cerr << "End of NetInterface::getConnection, thread id is " << pthread_self() << endl;
  return retval;
}

  
void
NetInterface::sendMulticast( bool enable, const char * clientName ) {
  cerr << "In NetInterface::sendMulticast, thread id is " << pthread_self() << endl;
  // If multicast isn't enabled, do nothing.
  if ( enableMulticast == false && enable == true )
    return;

  // Build a disconnect multicast message.
  Multicast m(false);
  m.setDisconnect( true );
  m.finish();

  // Send it
  sendDataToClient( clientName, m );
  cerr << "End of NetInterface::sendMulticast, thread id is " << pthread_self() << endl;
}

void
NetInterface::getMulticastGroup( char * &group, int &port, int &ttl ) {
  cerr << "In NetInterface::getMulticastGroup, thread id is " << pthread_self() << endl;
  static int currentPort = DEFAULT_MULTICAST_PORT;

  group = strdup( DEFAULT_MULTICAST_GROUP );
  port = currentPort++;
  ttl = DEFAULT_MULTICAST_TTL;
  cerr << "End of NetInterface::getMulticastGroup, thread id is " << pthread_self() << endl;
}

void
NetInterface::multicastCallback( MessageData *md ) {
  cerr << "In NetInterface::multicastCallback, thread id is " << pthread_self() << endl;
   
  if(md == NULL){
    cerr << "ERROR: NetInterface::multicastCallback - MessageData is NULL" << endl;
    return;
  }
  char buffer[ 1000 ];

  Multicast * mm = (Multicast *)(md->message);
  if(mm == NULL){
    cerr << "ERROR: NetInterface::multicastCallback - Multicast message is NULL" << endl;
    return;
  }


  // If the message is a 'disconnect', remove the client from the
  // multicast group (setting them back to PTP if necessary)
  if ( mm->isDisconnect() ) {
    cerr << "Message is a multicast disconnect" << endl;
    char * group;
    int port;
    int ttl;
    mm->getParams( group, port, ttl );
    snprintf( buffer, 1000,
	      "Client %s disconnected from group %s:%d", md->clientName,
	      group, port );
    Log::log( Logging::MESSAGE, buffer );
	      
    
    list<multicastGroup*>::iterator i;
    for ( i = multicastClients.begin(); i != multicastClients.end(); i++ ) {
      if ( (*i)->removeClient( md->clientName ) ) {
	// If we have no more multicast-enabled clients in this group,
	// disband it.
	if ( (*i)->clientNames.size() == 0 ) {
	  multicastClients.erase( i );
	  snprintf( buffer, 1000, "Deleting multicast group %s:%d",
		    (*i)->name, (*i)->port );
	  Log::log( Logging::MESSAGE, buffer );
	}
	return;
      }
    }
    
    snprintf( buffer, 1000,
	      "Non-multicast client %s tried to remove itself from a multicast group", md->clientName );
    Log::log( Logging::MESSAGE, buffer );
  }
  else if ( mm->isOkay() ) {
    cerr << "Message is an okay is join multicast" << endl;
    // Else if the message is a 'Yes', move that client from a multicast-
    // possible list to the appropriate multicast group.

        
    list<multicastGroup *>::iterator j;
    
    
    j = multicastClients.begin();
    for ( j = multicastClients.begin(); j != multicastClients.end(); j++ ) {
      if ( (*j)->switchClient( md->clientName ) ) {
	snprintf( buffer, 1000,
		  "Switching client %s to multicast group %s:%d",
		  md->clientName, (*j)->name, (*j)->port );
	Log::log( Logging::DEBUG, buffer );
	return;
      }
    }
   
    if ( j == multicastClients.end() ) {
      cerr << "Non-multicast client tried to add itself to a multicast group" << endl;
      // No relevent multicast groups exist. Tell the client to
      // disconnect (bugger off!)
      sendMulticast( false, md->clientName );
      return;
    }   

    snprintf( buffer, 1000,
	      "Non-multicast client %s tried to add itself to a multicast group", md->clientName );
    Log::log( Logging::MESSAGE, buffer ); 
    
  }
  else if ( mm->isRequest() ) {
    cerr << "Message is a request" << endl;
    // Else if the message is a 'No', update the 'answered' flag in the
    // multicast possible list.
    
    list<multicastGroup*>::iterator i;
    multicastPossible * mp = NULL;
    for ( i = multicastClients.begin(); i != multicastClients.end(); i++ ) {
      if ( (mp = (*i)->hasClient( md->clientName ) ) != NULL ) {
	mp->answered = true;
	snprintf( buffer, 1000, "Client %s refused to join multicast group",
		  md->clientName );
	Log::log( Logging::MESSAGE, buffer );
	return;
      }
    }
    
    snprintf( buffer, 1000,
	      "Non-multicast client %s tried to refuse to join a multicast group", md->clientName );
    Log::log( Logging::MESSAGE, buffer );  
  }
  else {
    cerr << "ERROR: NetInterface::multicastCallback - Bad message response to multicast" << endl;
    // Else - this is an error. Note that in the log.
    Log::log( Logging::ERROR, "Bad message response to multicast" );
  }
  cerr << "End of NetInterface::multicastCallback, thread id is " << pthread_self() << endl;
}

}  
}

//
// 8/30/2002
// - Commented out PTP send in sendDataToGroup
//
//
// 8/28/2002
// - Added some debugging code -- output size of test lists
//
