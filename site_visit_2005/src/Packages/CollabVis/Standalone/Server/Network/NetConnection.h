/*
 *
 * NetConnection: Abstraction for a network connection.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#ifndef __NetConnection_h_
#define __NetConnection_h_

#include <Network/PTPConnection.h>
#include <Network/dataItem.h>

#include <Thread/Runnable.h>
#include <Thread/Mailbox.h>
#include <Thread/CrowdMonitor.h>

#include <Logging/Log.h>

#include <vector>

namespace SemotusVisum {
namespace Network {

using namespace SCIRun;

enum {
  READ_ONLY,
  WRITE_ONLY,
  READ_WRITE
};

class NetConnection;

/**************************************
 
CLASS
   NetConnectionReader
   
KEYWORDS
   Network
   
DESCRIPTION

   NetConnectionReader is a helper to a network connection. It performs
   blocking network actions (read), and should thus be run in
   a different thread than its parent NetConnection.
   
****************************************/
class NetConnectionReader : public Runnable {
public:

  //////////
  // Constructor. Initializes our parent.
  NetConnectionReader( NetConnection *parent );

  //////////
  // Destructor. Cleans up our resources.
  ~NetConnectionReader();

  //////////
  // Run method, which waits for reads and writes to our connection.
  virtual void   run();

  //////////
  // Die now variable - used to cleanly exit run() method.
  bool dieNow;
  
  // 'Done running' variable.
  bool done;
  
protected:
  // Pointer to our parent
  NetConnection * parent;

};

/**************************************
 
CLASS
   NetConnectionWriter
   
KEYWORDS
   Network
   
DESCRIPTION

   NetConnectionWriter is a helper to a network connection. It performs
   blocking network actions (write), and should thus be run in
   a different thread than its parent NetConnection.
   
****************************************/
class NetConnectionWriter : public Runnable {
public:

  //////////
  // Constructor. Initializes our parent.
  NetConnectionWriter( NetConnection *parent );

  //////////
  // Destructor. Cleans up our resources.
  ~NetConnectionWriter();

  //////////
  // Run method, which waits for reads and writes to our connection.
  virtual void   run();

  //////////
  // Die now variable - used to cleanly exit run() method.
  bool dieNow;

  // 'Done running' variable.
  bool done;
  
protected:
  // Pointer to our parent
  NetConnection * parent;

  
  
};

/**************************************
 
CLASS
   NetMonitor
   
KEYWORDS
   Network
   
DESCRIPTION

   NetMonitor both disposes of NetConnections that are no longer active
   (ie, have disconnected), and notifies NetConnections when they have
   data ready to read. Furthermore, it also handles callbacks for
   incoming client handshake data. 
   
****************************************/

class NetMonitor : public Runnable {
public:

  //////////
  // Constructor. Gives us a ref to a list of net connections, as well
  // as a lock for that list.
  NetMonitor( list<NetConnection*>& connectList,
	      CrowdMonitor & connectListLock);

  //////////
  // Destructor. 
  ~NetMonitor();

  //////////
  // Run method, which checks both the mailboxs and the network for events.
  virtual void run();

  //////////
  // Returns a reference to our remove mailbox.
  inline Mailbox<NetConnection*>& getMailbox() {
    return removeBox;
  }
  
protected:
  Mailbox<NetConnection*> removeBox;     // Net connections to remove.
  list<NetConnection*> &connectList;     // Connection list to monitor 
  CrowdMonitor         &connectListLock; // Lock for connection list
};


/**************************************
 
CLASS
   NetConnection
   
KEYWORDS
   Network
   
DESCRIPTION

   NetConnection is an interface to a network connection. 
   
****************************************/

const int MAX_UNAVAILABLE   = 5;    // Max # times a connection is unavailable
                                    // before we consider it disconnected.
const int MAX_PENDING_ITEMS = 20;   // Max # items in our mailbox

const int PRIORITY_BIN_SIZE = 5;    // # slots in mailbox reserved for priority
                                    // messages.
const int MAX_MSG_SIZE      = 2048; // Max size of a client->server message

class NetConnection {
  friend class NetMonitor;
  friend class NetConnectionReader;
  friend class NetConnectionWriter;
public:

  //////////
  // Constructor. Initializes our connection, as well as the name of the
  // client that is initiating this connection.
  NetConnection( Connection &connection, 
		 const char * name, int flags=READ_WRITE );


  //////////
  // Copy constructor. Copies the name and connection, but creates a new
  // mailbox (as they can't be shared)...
  NetConnection( const NetConnection& netConnect);


  //////////
  // Tests for equality.
  bool operator==( const NetConnection& netConnect );
  
  //////////
  // Destructor. Deallocates all memory.
  ~NetConnection();

  //////////
  // Sets the (optional) nickname of this connection
  inline void setNickname( const char * nick ) {
    if ( !nick ) return;
    if ( nickname ) delete nickname;
    nickname = strdup( nick );
  }

  //////////
  // Returns the (optional) nickname of this connection.
  inline char * getNickname() { return nickname; }
  
  //////////
  // Returns the mailbox for this network connection.
  inline Mailbox<dataItem>& getMailbox() {
    //Logging::Log::log( Logging::DEBUG, "Grabbing mailbox!" );
    return mbox;
  }

  //////////
  // Returns the name of the client initiating this connection.
  inline  char * getName() {
    return name;
  }

  //////////
  // Returns the low-level connection.
  inline Connection& getConnection() {
    return connection;
  }

  //////////
  // Returns the list of unsent priority messages.
  inline vector<dataItem> getPriorityList() {
    return priorityList;
  }
  
  //////////
  // Turns on/off use of the network dispatch manager. Adds an optional
  // extra function to be called when we get data.
  static inline void useDM( bool use,  void (*func)(void *) = NULL ) {
    NetConnection::useDispatchManager = use;
    NetConnection::callbackFunction = func;
  }

  static NetMonitor& getNetMonitor();

  //////////
  // Callback for 'goodbye' messages.
  static void goodbyeCallback( void * obj, MessageData *input );
  
protected:

  // Nickname
  char * nickname;
  
  // Thread for network monitor. 
  static Thread *incomingThread;

  // List of all active connections.
  // static list<NetConnection*> connectionList;

  // Lock to ensure that we don't munge our connection list.
  //static CrowdMonitor connectionListLock;

  // Monitors all sockets for incoming connections.
  static NetMonitor *netMonitor;

  // True if we should use the net dispatch manager for inbound data.
  static bool useDispatchManager;

  // Callback function to call if we are not using the dispatch manager.
  static void (*callbackFunction)(void *);
  
  // Sends messages to all net connections who have data available.
  // NOTE - THIS FUNCTION BLOCKS!
  //static void notifyDataAvailable();
  
  // Writes data to the network.
  int                 write( const char * data, const int numBytes );

  // Name of the client on this connection.
  char                *name;

  // Underlying low-level connection
  Connection          &connection;

  // Mailbox for incoming/outgoing data
  Mailbox<dataItem>   mbox;

  // Was this a clean disconnect (ie, we got a goodbye message)?
  bool                cleanDisconnect;
  
  // Thread for Reader
  Thread * readThread;

  // Thread for Writer
  Thread * writeThread;
  
  // Reader for multithreading
  NetConnectionReader * Reader;

  // Writer for multithreading
  NetConnectionWriter * Writer;

  pid_t readerPID;

  // List to hold priority items
  vector<dataItem>    priorityList;
};


}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:25  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:44  simpson
// Adding CollabVis files/dirs
//
// Revision 1.10  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.9  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.8  2001/07/16 20:29:36  luke
// Updated network stuff...
//
// Revision 1.7  2001/06/07 22:26:31  luke
// Added multicast net interface driver, fixed and enhanced multicast code
//
// Revision 1.6  2001/05/28 15:08:24  luke
// Revamped networking. Now we have different reading and writing threads for
// each connection. We also have a separate thread to get handshake data. Other
// minor improvements.
//
// Revision 1.5  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.4  2001/05/12 03:29:11  luke
// Now uses messages instead of XML. Also moved drivers to new location
//
// Revision 1.3  2001/05/01 20:55:55  luke
// Works for a single client, but client disconnect causes the server to seg fault
//
// Revision 1.2  2001/04/12 20:14:47  luke
// Cleaned some bugs, moved error detection to PTPConnection. Some bugs still remain.
//
// Revision 1.1  2001/04/11 17:47:24  luke
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
