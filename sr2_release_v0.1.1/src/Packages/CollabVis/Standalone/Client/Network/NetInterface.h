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



namespace SemotusVisum {

using namespace SCIRun;

/** Default server port to connect to */
const int          DEFAULT_SERVER_PORT     = 6210;

/** Default multicast group */
const char * const DEFAULT_MULTICAST_GROUP = "228.6.6.6";

/** Default multicast port */
const int          DEFAULT_MULTICAST_PORT  = DEFAULT_SERVER_PORT;

/** Default multicast time-to-live */
const int          DEFAULT_MULTICAST_TTL   = 5;

/** Callback function type for client connects. */
typedef void (*connectFunc)( void * obj, const char * name );

/** Transfer Modes */
static const char * const transferMethods[] = { "PTP", 
						"Reliable Multicast",
						"IP Multicast",
						"None",
						0 };

/**
 * NetInterface provides an interface for connecting to a server, 
 * receiving data from the server, and sending data to the server.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetInterface {
  friend class NetConnection;
  friend class NetMonitor;
public:

  /**
   * Connects to the multicast group with the given port
   *
   * @param group   Multicast group
   * @param port    Port number
   */
  void connectToMulticast( string group, int port=DEFAULT_MULTICAST_PORT );

  
  /**
   * Connects to the server with the given name on the given port
   *
   * @param server        Server name, IP addr or machine name.
   * @param port          Port number
   * @return              True if the connection succeeds; else false.
   */
  bool connectToServer( string server, int port=DEFAULT_SERVER_PORT );
  
  /**
   *  Disconnects from the multicast group.
   *
   */
  void disconnectFromMulticast();
  
  /**
   *  Disconnects from the server
   *
   */
  void disconnectFromServer();
  
  /**
   * Uses message to determine if multicast should be enabled.
   *
   * @param message       Multicast message from server.
   */
  void enableMulticast( const Multicast &message );

  /**
   * Uses message to determine appropriate data transfer mode.
   *
   * @param message       Transfer message from server.
   */
  void setTransferMode( MessageData * input );
  
  /**
   * Returns numBytes of data from the multicast group.
   *
   * @param numBytes      Number of bytes to read.
   * @return              Newly allocated data from network, or NULL on error.
   */
  char * getDataFromMulticast( int &numBytes );
  
  /**
   * Returns numBytes of data from the server.
   *
   * @param numBytes     Number of bytes to read.
   * @return             Newly allocated data from network, or NULL on error.
   */
  char * getDataFromServer( int &numBytes );
  
  /**
   * Returns true if multicast is enabled.
   *
   * @return True if multicast is enabled (but not necessarily connected to a
   *         multicast group!).
   */
  inline bool isMulticastEnabled() const { return multicastEnabled; }

  
  /**
   * Sends the given data to the server
   *
   * @param data      Data to send.
   * @param numBytes  Number of bytes to send.
   */
  void sendDataToServer( const string& data, const int numBytes );
  
  /**
   * Sends the given message to the server
   *
   * @param m     Message to send.
   */
  inline void sendDataToServer( MessageBase *m ) {
    if ( m != NULL ) {
      m->finish();
      sendDataToServer( m->getOutput(), m->getOutput().length() );
    }
    else
      Log::log( WARNING, "NULL message sent to server!" );
  }
  
  /**
   * Sets if multicast is enabled.
   *
   * @param enabled       True to enable multicast - false to disable it.
   */
  inline void setMulticastEnabled( const bool enabled ) {
    multicastEnabled = enabled;
  }
  
  /**
   * Returns the name of the machine.
   *
   * @return String name of the machine.
   */
  static inline const char * getMachineName() { return MyMachineName(); }
  
  /**
   *  Returns a reference to the singleton instance of the network
   * interface
   *
   */
  static NetInterface&  getInstance();

  /**
   * Data marker for the end of XML data and the beginning of raw data. 
   */
  static const char * const dataMarker;  

  /**
   * Static callback to enable multicast. 
   *
   * @param obj    Object from net dispatch manager - should be net interface.
   * @param input  Message input.
   */
  static void __enableMulticast( void * obj, MessageData * input ) {
    Multicast m = *((Multicast *)(input->message));
    cerr << "In __enableMulticast - group = " << m.getGroup() << ", port = " << m.getPort() << endl; 
    ((NetInterface *)obj)->enableMulticast( *((Multicast *)input->message) );
  }

  /**
   * Static callback for data transfer mode message. 
   *
   * @param obj    Object from net dispatch manager - should be net interface.
   * @param input  Message input.
   */
  static void __transferCallback(void * obj, MessageData * input) {
    Log::log( DEBUG, "Calling data transfer mode callback" ); 
    //Transfer *t = (Transfer *)(input->message);
    ((NetInterface *)obj)->setTransferMode(input);
   
  }


protected:

  /** The singleton network interface. */
  static NetInterface net;               

  /** Is multicast enabled? */
  bool                  multicastEnabled; 

  /** Connection to server. */
  NetConnection         *serverConnection;

  /** Connection to multicast */
  NetConnection         *multicastConnection;

  /** Data transfer mode for multiple clients  (i.e. PTP, reliable multicast, or IP Multicast) */
  string  transferMode;        

  /** IDs for callbacks */
  int handshakeID, multicastID, transferID;

 
  
  /**
   *  Constructor. Can't create or destroy them.
   *
   */
  NetInterface();
  
  /**
   *  Destructor. Can't create or destroy them.
   *
   */
  ~NetInterface();
  
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:32  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:01  simpson
// Adding CollabVis files/dirs
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
