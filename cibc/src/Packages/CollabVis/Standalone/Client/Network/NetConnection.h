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

using namespace SCIRun;

/** Enumerations for reading and writing */
enum {
  READ_ONLY,
  WRITE_ONLY,
  READ_WRITE
};

class NetConnection;

/**
 * NetConnectionReader is a helper to a network connection. It performs
 * blocking network actions (read), and should thus be run in
 * a different thread than its parent NetConnection.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetConnectionReader : public Runnable {
public:

  /**
   * Constructor. Initializes our parent.
   *
   * @param parent        Parent net connection.
   */
  NetConnectionReader( NetConnection *parent );

  /**
   * Destructor. Cleans up our resources.
   *
   */
  ~NetConnectionReader();

  /**
   * Run method, which waits for reads from our connection.
   *
   */
  virtual void   run();

  
  /// Die now variable - used to cleanly exit run() method.
  bool dieNow;
  
  /// 'Done running' variable.
  bool done;
  
protected:
  /// Pointer to our parent
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
/**
 * NetConnectionWriter is a helper to a network connection. It performs
 * blocking network actions (write), and should thus be run in
 * a different thread than its parent NetConnection.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetConnectionWriter : public Runnable {
public:

  /**
   * Constructor. Initializes our parent.
   *
   * @param parent        Parent net connection.
   */
  NetConnectionWriter( NetConnection *parent );

  /**
   * Destructor. Cleans up our resources.
   *
   */
  ~NetConnectionWriter();

  /**
   * Run method, which waits for writes to our connection.
   *
   */
  virtual void   run();

  
  /// Die now variable - used to cleanly exit run() method.
  bool dieNow;

  /// 'Done running' variable.
  bool done;
  
protected:
  /// Pointer to our parent
  NetConnection * parent;

};

/**
 * NetMonitor both disposes of NetConnections that are no longer active
 * (ie, have disconnected).
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetMonitor : public Runnable {
public:

  /**
   * Constructor.
   *
   */
  NetMonitor() : removeBox("NMRemoveBox", 2 ) {}
  
  /**
   * Destructor. 
   *
   */
  ~NetMonitor();

  /**
   * Run method, which checks both the mailboxs and the network for events.
   *
   */
  virtual void run();

  /**
   * Returns a reference to our remove mailbox.
   *
   */
  inline Mailbox<NetConnection*>& getMailbox() {
    return removeBox;
  }
  
protected:
  /// Net connections to remove.
  Mailbox<NetConnection*> removeBox;     
};


/**************************************
 
CLASS
   NetConnection
   
KEYWORDS
   Network
   
DESCRIPTION

   NetConnection is an interface to a network connection. 
   
****************************************/

/** Max # times a connection is unavailable before we consider it
    disconnected.*/
const int MAX_UNAVAILABLE   = 5;

/** Max # items in our mailbox */
const int MAX_PENDING_ITEMS = 20;  

/** # slots in mailbox reserved for priority messages. */
const int PRIORITY_BIN_SIZE = 5;

/**
 * NetConnection is an interface to a nonblocking network connection. 
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetConnection {
  friend class NetMonitor;
  friend class NetConnectionReader;
  friend class NetConnectionWriter;
public:

  /**
   * Constructor. Initializes our connection, as well as the name of the
   * client (us) that is initiating this connection.
   *
   * @param connection    Blocking network connection
   * @param name          Connection name
   * @param flags         Read/Write flags
   */
  NetConnection( Connection &connection, 
		 const string &name, int flags=READ_WRITE );


  /**
   *  Copy constructor. Copies the name and connection, but creates a new
   *   mailbox (as they can't be shared)...
   *
   * @param netConnect    Connection to copy
   */
  NetConnection( const NetConnection& netConnect);


  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~NetConnection();

  
  /**
   * Tests for equality.
   *
   * @param netConnect    Connection to test.
   */
  bool operator==( const NetConnection& netConnect );
  
  /**
   * Returns the mailbox for this network connection.
   *
   * @return    I/O request mailbox
   */
  inline Mailbox<dataItem>& getMailbox() {
    return mbox;
  }

  /**
   *  Returns the name of this connection.
   *
   * @return   Name of the connection
   */
  inline string& getName() {
    return name;
  }

  /**
   *  Returns the low-level (blocking) connection.
   *
   */
  inline Connection& getConnection() {
    return connection;
  }

  /**
   * Explicitly reads data (blocking) from the network. Typically not used.
   *
   * @param data          Preallocated storage for data
   * @param numBytes      Number of bytes to read.
   * @return              Number of bytes actually read.
   */
  inline int read( char * data, int numBytes ) {
    return connection.read( data, numBytes );
  }
  
  /**
   * Explicitly writes data (blocking) to network. 
   *
   * @param data         Data to write
   * @param numBytes     Number of bytes of data to write
   * @return             Number of bytes actually written
   */
  inline int write( const char * data, int numBytes ) {
    unsigned dataSize = (unsigned)numBytes;
    DataDescriptor dd = SIMPLE_DATA( UNSIGNED_INT_TYPE, 1 );
    ConvertHostToNetwork( (void *)&dataSize,
			  (void *)&dataSize,
			  &dd,
			  1 );
    if ( connection.write( (const char *)&dataSize,
			   sizeof(unsigned int) ) < 0 ) {
      Log::log( ERROR, "Error writing data size" );
      return -1;
    }
    return connection.write( data, numBytes );
  }

  /**
   * Closes the connection.
   *
   */
  inline void close() {
    connection.close();
  }
  
  /**
   * Turns on/off use of the network dispatch manager. Adds an optional
   * extra function to be called when we get data. Only used for testing.
   *
   * @param use     True to turn on dispatch manager; else false
   * @param func    Function to call if DM not being used.
   */
  static inline void useDM( bool use,  void (*func)(void *) = NULL ) {
    NetConnection::useDispatchManager = use;
    NetConnection::callbackFunction = func;
  }

  /**
   * Returns a reference to the network monitor.
   *
   * @return  Reference to network monitor.
   */
  static inline NetMonitor& getNetMonitor() {
    return netMonitor;
  }

protected:

  /// Thread for network monitor. 
  static Thread *incomingThread;

  /// Monitors connections to see if they should be removed.
  static NetMonitor netMonitor;

  /// True if we should use the net dispatch manager for inbound data.
  static bool useDispatchManager;

  /// Callback function to call if we are not using the dispatch manager.
  static void (*callbackFunction)(void *);
  
  /// Name of this connection.
  string                name;

  /// Underlying low-level connection
  Connection          &connection;

  /// Mailbox for incoming/outgoing data
  Mailbox<dataItem>   mbox;

  /// Was this a clean disconnect?
  bool                cleanDisconnect;
  
  /// Thread for Reader
  Thread * readThread;

  /// Thread for Writer
  Thread * writeThread;
  
  /// Reader for multithreading
  NetConnectionReader * Reader;

  /// Writer for multithreading
  NetConnectionWriter * Writer;

  /// Process ID for reader.
  pid_t readerPID;

};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:32  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:00  simpson
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
