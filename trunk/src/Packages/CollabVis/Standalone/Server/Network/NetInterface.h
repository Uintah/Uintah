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

#ifndef __NetInterface_h_
#define __NetInterface_h_

#include <list>
#include <vector>
#include <stdio.h>

#include <Network/PTPConnection.h>
#include <Network/MulticastConnection.h>
#include <Network/NetConnection.h>
#include <Network/NetDispatchManager.h>

#define ASIP_SHORT_NAMES
#include <Network/AppleSeeds/ipseed.h>

#define ASFMT_SHORT_NAMES
#include <Network/AppleSeeds/formatseed.h>

#include <Thread/Thread.h>
#include <Thread/Runnable.h>
#include <Thread/Mailbox.h>
#include <Thread/CrowdMonitor.h>
#include <Thread/Mutex.h>

#include <Malloc/Allocator.h>
#include <Logging/Log.h>

#include <Properties/ServerProperties.h>



namespace SemotusVisum {
namespace Network {

using namespace SCIRun;

class NetInterface;



////////////////
// Connection info for creating a data connection.
struct connectItem {
  int socket;
};


/**************************************
 
CLASS
   multicastPossible
   
KEYWORDS
   Network, Multicast
   
DESCRIPTION

   Data structure for clients with pending additions to a multicast
   group.
   
****************************************/
struct multicastPossible {
  
  //////////
  // Constructor - sets the name, group, port, and 'answered' flags.  
  multicastPossible( const char * name, const char * group, int port,
		     bool answered=false ) :
    answered(answered), port(port) {
    if ( name )
      clientName = strdup( name );
    else clientName = NULL;
    if ( group )
      this->group = strdup( group );
    else
      this->group = NULL;
  }

  //////////
  // Destructor - deallocates memory.
  ~multicastPossible() {
    delete clientName;
    delete group;
  }

  // Client name
  char * clientName;

  // True if they have responded.
  bool   answered;

  // Multicast group name
  char * group;

  // Multicast group port
  int port;
};

/**************************************
 
CLASS
   multicastGroup
   
KEYWORDS
   Network, Multicast
   
DESCRIPTION

   Data structure for clients belonging to a multicast group.
   
****************************************/
struct multicastGroup {
public:
  //////////
  // Constructor - sets the group name.
  multicastGroup( NetConnection * group, char * groupName, int port) :
    group( group ), name( NULL ), port( port ) {
    if ( groupName ) name = strdup( groupName );
    else name = NULL;
  }

  //////////
  // Destructor - deallocates all memory.
  ~multicastGroup() { clientNames.clear(); delete name; delete group; }

  //////////
  // Adds a client to the group
  inline void addClient( const char * name ) {
    if ( name )
      clientNames.push_front( strdup( name ) );
  }

  //////////
  // Adds a client to the 'not multicast (yet) but in this group' list.
  inline void addOtherClient( multicastPossible *mp ) {
    otherClients.push_front( mp );
  }

  //////////
  // Returns true if all clients in this group have answered; else,
  // returns false.
  inline bool allAnswered() {
    list<multicastPossible*>::iterator i;
    for ( i = otherClients.begin(); i != otherClients.end(); i++ )
      if ( (*i)->answered == false ) return false;
    return true;
  }
  
  //////////
  // Returns the messagePossible structure if the given client is in this
  // group and is still in PTP mode. Else, returns NULL.
  inline multicastPossible * hasClient( const char * clientName ) {
    list<multicastPossible*>::iterator i;
    for ( i = otherClients.begin(); i != otherClients.end(); i++ )
      if ( !strcmp( (*i)->clientName, clientName ) )
	return *i;
    return NULL;
  }

  
  //////////
  // Switches a client from the 'no multicast' portion of the group to
  // the 'multicast' list. Returns true if successful, else false.
  inline bool switchClient( const char * clientName ) {
    list<multicastPossible*>::iterator i;
    for ( i = otherClients.begin(); i != otherClients.end(); i++ )
      if ( !strcmp( (*i)->clientName, clientName ) ) {
	addClient( clientName );
	otherClients.erase( i );
	return true;
      }
    return false;
  }
  
  //////////
  // Removes a client from the group, and adds it to the list of possibles.
  inline bool removeClient( char * name ) {
    for ( list<char *>::iterator i = clientNames.begin();
	  i != clientNames.end(); i++ )
      if ( !strcmp( name, (char *)*i ) ) {
	clientNames.erase( i );
	otherClients.push_front( scinew multicastPossible( name,
							   this->name, port,
							   true ) );
	return true;
      }
    return false;
  }

  //////////
  // Removes a client from the group permanently.
  inline bool deleteClient( const char * name ) {
    for ( list<char *>::iterator i = clientNames.begin();
	  i != clientNames.end(); i++ )
      if ( !strcmp( name, (char *)*i ) ) {
	clientNames.erase( i );
	return true;
      }
    return false;
  }
  
  // Client names
  list<char *> clientNames;
  
  // Clients in this group but not (yet) on multicast
  list<multicastPossible*> otherClients;
  
  // Pointer to multicast connection
  NetConnection * group;

  // Multicast group name
  char * name;

  // Multicast group port
  int port;
};

//////////
// Default server port to listen on
const int          DEFAULT_SERVER_PORT     = 6210;

//////////
// Default multicast group
const char * const DEFAULT_MULTICAST_GROUP = "228.6.6.6";

//////////
// Default multicast port
const int          DEFAULT_MULTICAST_PORT  = DEFAULT_SERVER_PORT;

//////////
// Default multicast time-to-live
const int          DEFAULT_MULTICAST_TTL   = 5;

/**************************************
 
CLASS
   NetListener
   
KEYWORDS
   Network
   
DESCRIPTION

   NetListener listens on a socket. When a client connects, NetListener
   passes on the connection request to NetInterface and resumes listening.
   
****************************************/

class NetListener : public Runnable {
public:

  //////////
  // Constructor. Passes in a reference to the network interface and
  // an optional port.
  NetListener( NetInterface &net, int port );

  //////////
  // Destructor
  ~NetListener();

  //////////
  // Run method. Runs as a new thread, listening on the predetermined
  // port.
  virtual void run();

  
  /**
   * Returns the listening port
   *
   * @return Port that the listener is listening on.
   */
  int getPort() const { return port; }
  
  bool dieNow;  // Should we exit now?
protected:
  int port;                   // Port to listen to
  NetInterface &net;          // The network interface that owns us.
  Mailbox<connectItem>& mbox; // The mailbox that we put connection info
                              // into for the network interface.
  ASIP_Socket listenSocket;        // Listening socket.
};


/////////
// Callback function for client connects.
typedef void (*connectFunc)( void * obj, const char * name );

/**************************************
 
CLASS
   NetInterface
   
KEYWORDS
   Network
   
DESCRIPTION

   NetInterface provides an interface for listening for connections,
   receiving data from clients, and sending data to clients.
   
****************************************/

class NetInterface {
  friend class NetListener;
  friend class NetConnection;
  friend class NetMonitor;
public:

  ///////////
  // Initiates a listener on the given port. This call does not block...
  void   listen( int port=DEFAULT_SERVER_PORT );

  ///////////
  // Stops the network interface - shuts down all connections and stops
  // listening on the given port...
  void   stop();
  
  //////////
  // Initiates a listener for multicast connections on the given
  // port. NOT DONE.
  void   listenMulticast(int port);

  /////////
  // Adds the given function to a list of functions to be called every
  // time a client connects.
  void   onConnect( void * obj, connectFunc func );
  
  //////////
  // Enables/disables remote module requests. When disabled, it will respond
  // to client requests with 'Function not implemented'.
  void   enableRemoteModuleCallback( bool enable );

  //////////
  // Callback function for rejecting remote module requests.
  static void   remoteModuleCallback( void * obj, MessageData *input );
  
  //////////
  // Callback function for getting a handshake from a client.
  static void   getHandshake( void * obj, MessageData *input );

  //////////
  // Callback function for getting a chat message from a client.
  static void   getChat( void * obj, MessageData *input );

  //////////
  // Callback function for getting a getClientList message from a client.
  static void   getClientList( void * obj, MessageData *input );

  //////////
  // Callback function for getting a transfer message from a client.
  static void transferCallback( void * obj, MessageData *input );

  //////////
  // Callback function for getting a collaborate message from a client.
  static void   getCollaborate( void * obj, MessageData *input );
  


  //////////
  // Sends data to the given client. This call does not block. Returns
  // true if the client still exists; else returns false.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  bool   sendDataToClient(const char * clientName,
			  const char * data,
			  const DataTypes dataType,
			  const int numBytes,
			  bool  copy=true );

  //////////
  // Sends priority data to the given client. This call does not block. Returns
  // true if the client still exists; else returns false.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  bool   sendPriorityDataToClient(const char * clientName,
				  const char * data,
				  const DataTypes dataType,
				  const int numBytes,
				  bool  copy=true );

  //////////
  // Sends message to the given client. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  inline bool sendDataToClient( const char * clientName,
				MessageBase &message ) {
    return sendDataToClient( clientName,
			     message.getOutput(),
			     CHAR_TYPE,
			     strlen( message.getOutput() ),
			     true ); // True, as the message deletes its text
  }

  //////////
  // Sends priority message to the given client. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  inline bool sendPriorityDataToClient( const char * clientName,
					MessageBase &message ) {
    return sendPriorityDataToClient( clientName,
				     message.getOutput(),
				     CHAR_TYPE,
				     strlen( message.getOutput() ),
				     true ); // True, as the message deletes
                                             // its text
  }
  
  /////////////
  // Sends the data to the given clients.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  bool sendDataToClients( list<char *>  &clients,
			  const char * data,
			  const DataTypes dataType,
			  const int numBytes,
			  bool copy=true);
  
  /////////////
  // Sends the priority data to the given clients.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  bool sendPriorityDataToClients( list<char *>  &clients,
				  const char * data,
				  const DataTypes dataType,
				  const int numBytes,
				  bool copy=true);
  

  /////////////
  // Sends the message to the given clients.
  // The copy parameter indicates that the function should create its own
  // copy of the data. 
  inline bool sendDataToClients( list<char *>  &clients,
				 MessageBase &message ) {
    return sendDataToClients( clients,
			      message.getOutput(),
			      CHAR_TYPE,
			      strlen( message.getOutput() ),
			      true ); // True, as the message deletes its text
  }

  /////////////
  // Sends the message to the given clients.
  // The copy parameter indicates that the function should create its own
  // copy of the data. 
  inline bool sendPriorityDataToClients( list<char *>  &clients,
					 MessageBase &message) {
    return sendPriorityDataToClients( clients,
				      message.getOutput(),
				      CHAR_TYPE,
				      strlen( message.getOutput() ),
				      true ); // True, as the message deletes
                                              // its text
  }
  
  
  //////////
  // Sends data to all clients. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  void   sendDataToAllClients(const char * data,  
			      const DataTypes dataType, 
			      const int numBytes,
			      bool copy=true );

  //////////
  // Sends priority data to all clients. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  void   sendPriorityDataToAllClients(const char * data,  
				      const DataTypes dataType, 
				      const int numBytes,
				      bool copy=true ); 
  
  //////////
  // Sends message to all clients. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  inline void sendDataToAllClients( MessageBase &message ) {  
    sendDataToAllClients( message.getOutput(),
			  CHAR_TYPE,
			  strlen( message.getOutput() ),
			  true ); // True, as the message deletes its text
  }

  //////////
  // Sends priority message to all clients. This call does not block.
  // The copy parameter indicates that the function should create its own
  // copy of the data.
  inline void sendPriorityDataToAllClients( MessageBase &message ) {  
    sendDataToAllClients( message.getOutput(),
			  CHAR_TYPE,
			  strlen( message.getOutput() ),
			  true ); // True, as the message deletes its text
  } 
  
  
  //////////
  // Creates and returns a multicast group.
  multicastGroup * createMulticastGroup( list<char *> &clients,
					 const char * group=NULL,
					 const int port=-1);

  //////////
  // Adds a client to a multicast group.
  void addToMulticastGroup( const char * client, multicastGroup * mg );

  //////////
  // Removes a client from this multicast group.
  void deleteFromMulticastGroup( const char * client, multicastGroup *mg );
  
  //////////
  // Sends data to the given group
  bool sendDataToGroup( const multicastGroup *mg, const char * data,
			const DataTypes dataType, const int numBytes );

  //////////
  // Sends message to the given group
  inline bool sendDataToGroup( const multicastGroup *mg,
			       MessageBase &message ) {  
    return sendDataToGroup( mg,
			    message.getOutput(),
			    CHAR_TYPE, 
			    strlen( message.getOutput() ) ); 
  } 
  
  //////////
  // Returns true if the given multicast group is still valid; else
  // returns false. If it discovers an invalid group in the list, the
  // group will be destroyed.
  inline bool validGroup( multicastGroup *mg ) {
    list<multicastGroup*>::iterator i;
    for ( i = multicastClients.begin(); i != multicastClients.end(); i++ )
      if ( mg == *i ) {
	if ( mg->clientNames.size() != 0 || !mg->allAnswered() )
	  return true;
	if ( mg->clientNames.size() == 0 && mg->allAnswered() ) {
	  multicastClients.erase( i );
	  return false;
	}
      }
    return false;
  }
  
  
  //////////
  // Returns a reference to the singleton instance of the network
  // interface
  static NetInterface&  getInstance();

  //////////
  // Blocks until we have at least one client connection. Returns the
  // name of the last client to connect.
  char *   waitForConnections();

  //////////
  // Returns a freshly allocated list (nothing aliased) of client names.
  // Returns NULL if no clients connected.
  list<char *>*   getClientNames();

  //////////
  // Returns the number of clients currently connected.
  inline int clientsConnected() {
    int result;
    netConnectionLock.readLock();
    result = clientConnections.size();
    netConnectionLock.readUnlock();
    return result;
  }
  
  //////////
  // Returns true if there are new connections since this function
  // or waitForConnections was last called.
  inline bool newConnections() {
    if ( newConnects ) {
      newConnects = false;
      return true;
    }
    return false;
  }

  //////////
  // Enables/disables automatic multicast
  inline void setMulticast( bool enable ) { enableMulticast = enable; }

  //////////
  // Returns true if automatic multicast is enabled. 
  inline bool getMulticast() const { return enableMulticast; }

  //////////
  // Callback for client responses to multicast messages.
  static void multicastResponseCallback( void * obj, MessageData *md ) {
    if ( obj ) ((NetInterface *)obj)->multicastCallback( md );
  }
  
  //////////
  // Data marker for the end of XML data and the beginning of raw data.
  static const char * const dataMarker; 

  //////////
  // Removes and deletes all connections, PTP and multicast
  void   removeAllConnections();
  
  void   removeConnection( NetConnection * nc );

  //////////
  // Looks up and returns the net connection with the given name. If
  // the given client does not exist, it returns NULL.
  NetConnection * getConnection( const char * name );

  ////////////
  // Notifies other clients that this client has changed rendering groups.
  void   modifyClientGroup( const char * clientName,
			    const char * group );

  ///////////
  // Returns the string name of the current data transfer mode
  inline string getTransferMode(){ return transferMode; }

  

protected:
  
  list<NetConnection*> clientConnections;    // Client network connections
  list<MulticastConnection*>  multicastConnections;  // Multicast connections.

  Mailbox<connectItem> connectMBox;          // Mailbox for network connection
                                             // communication.
  
  NetListener          *listener;            // Listener for new connections.
  CrowdMonitor          netConnectionLock;   // Monitor for network connection
                                             // list, so we don't munge our
                                             // data structures.
  
  static NetInterface  *net;                 // The singleton network
                                             // interface.

  Mutex                 haveConnections;     // Allows blocking until
                                             // we have a connection.

  bool                  newConnects;         // Are there new connections?

  int                   remoteModuleID;      // Remote module callback ID.
  bool                  enableMulticast;     // Is automatic multicasting
                                             // enabled?

  string                transferMode;        // Data transfer mode for multiple clients 
                                             // (i.e. PTP, reliable multicast, or IP Multicast)

  list<multicastGroup*> multicastClients;    // Groups of clients currently
                                             // connected to multicast groups.

  vector<connectFunc>   connectFunctions;    // Functions to be called whenever
                                             // a client connects.
  vector<void *>        connectData;         // Objects to be passed to the
                                             // above callbacks.

  Thread * listenThread;                     // Thread for listener.

  list<multicastGroup*> testMulticastClients5; // test list for debugging
  
  void   addConnection( Connection &c, const char * hostname );
  

  // Sends a message to clients informing them of a switch to/from multicast.
  void   sendMulticast( bool enable, const char * clientName );

  // Finds a new multicast group and port and ttl
  void   getMulticastGroup( char * &group, int &port, int &ttl );

  // Real (non-wrapper) multicast callback.
  void   multicastCallback( MessageData *md );

  // Real function to send data to client.
  bool   realSendDataToClient( const char * clientName,
			       const char * data,
			       const DataTypes dataType,
			       const int numBytes,
			       bool copy,
			       bool priority );

  // Real function to send data to clients.
  bool   realSendDataToClients( list<char *>  &clients,
				const char * data,
				const DataTypes dataType,
				const int numBytes,
				bool  copy,
				bool  priority);

  // Real function to send data to all clients.
  void realSendDataToAllClients(const char * data,
				const DataTypes dataType,
				const int numBytes,
				bool copy,
				bool priority );
  
  // Can't create or destroy them.
  NetInterface();
  ~NetInterface();
  
};

}
}
#endif



//
// $Log$
// Revision 1.1  2003/07/22 15:46:26  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:46  simpson
// Adding CollabVis files/dirs
//
//
// 8/29/2002 -- added more debugging code and temporarily took multicastClients out of protected section so that its value can be examined more easily 
// - Moved multicastClients back to protected section
// 
// 8/28/2002
// Added some debugging code -- extra lists to test for memory bug
//
// Revision 1.29  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.28  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.27  2001/10/04 16:55:01  luke
// Updated XDisplay to allow refresh
//
// Revision 1.26  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.25  2001/08/20 17:46:34  luke
// Net interface can now be stopped
//
// Revision 1.24  2001/08/20 15:41:30  luke
// Made connect/disconnect code more robust by using Thread::interrupt()
//
// Revision 1.23  2001/08/08 01:58:05  luke
// Multicast working preliminary on Linux
//
// Revision 1.22  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.21  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.20  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.19  2001/07/16 20:29:36  luke
// Updated network stuff...
//
// Revision 1.18  2001/06/07 22:26:31  luke
// Added multicast net interface driver, fixed and enhanced multicast code
//
// Revision 1.17  2001/06/06 21:16:44  luke
// Callback functions now an option in NetDispatchManager
//
// Revision 1.16  2001/06/05 17:44:57  luke
// Multicast basics working
//
// Revision 1.15  2001/05/31 17:32:46  luke
// Most functions switched to use network byte order. Still need to alter Image Renderer, and some drivers do not fully work
//
// Revision 1.14  2001/05/29 03:09:45  luke
// Linux version mostly works - preparing to integrate IRIX compilation changes
//
// Revision 1.13  2001/05/28 15:08:24  luke
// Revamped networking. Now we have different reading and writing threads for
// each connection. We also have a separate thread to get handshake data. Other
// minor improvements.
//
// Revision 1.12  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.11  2001/05/17 19:38:58  luke
// New network data reading code.
//
// Revision 1.10  2001/05/14 22:39:02  luke
// Fixed compiler warnings
//
// Revision 1.9  2001/05/14 20:08:50  luke
// Added block for client & client name list features
//
// Revision 1.8  2001/05/14 18:07:56  luke
// Finished documentation
//
// Revision 1.7  2001/05/12 03:29:11  luke
// Now uses messages instead of XML. Also moved drivers to new location
//
// Revision 1.6  2001/05/03 18:18:41  luke
// Writing from server to client works. Server sends handshake upon client connection.
//
// Revision 1.5  2001/05/01 20:55:55  luke
// Works for a single client, but client disconnect causes the server to seg fault
//
// Revision 1.4  2001/04/11 17:47:25  luke
// Net connections and net interface work, but with a few bugs
//
// Revision 1.3  2001/04/05 22:28:01  luke
// Documentation done
//
// Revision 1.2  2001/04/04 21:45:29  luke
// Added NetDispatch Driver. Fixed bugs in NDM.
//
// Revision 1.1  2001/02/08 23:53:29  luke
// Added network stuff, incorporated SemotusVisum namespace
//
