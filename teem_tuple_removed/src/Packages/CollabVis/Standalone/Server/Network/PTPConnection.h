/*
 *
 * PTPConnection: Provides a point-to-point network connection
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#ifndef __PTPConnection_h_
#define __PTPConnection_h_

#include <list>

#include <Network/Connection.h>
#define ASIP_SHORT_NAMES
#include <Network/AppleSeeds/ipseed.h>

#define ASFMT_SHORT_NAMES
#include <Network/AppleSeeds/formatseed.h>

#include <Thread/CrowdMonitor.h>
namespace SemotusVisum {
namespace Network {

//using namespace SemotusVisum::Thread;
using namespace SCIRun;
using namespace std;

/**************************************
 
CLASS
   PTPConnection
   
KEYWORDS
   Network, Connection, Point-to-point
   
DESCRIPTION

   PTPConnection implements a blocking point-to-point connection. It
   provides safe reads and writes to/from the network.
   
****************************************/
class PTPConnection : public Connection {
public:

  //////////
  // Constructor
  PTPConnection();

  //////////
  // Constructor
  PTPConnection( Socket theSocket );
  
  //////////
  // Destructor
  virtual ~PTPConnection();

  //////////
  // Reads the given number of bytes from the network, and stores it
  // into the memory pointed to by data. Returns the number of bytes read.
  // NOTE - THIS FUNCTION BLOCKS!
  virtual int read ( char * data, int numBytes );

  //////////
  // Writes the given number of bytes from the memory pointed to by data
  // to the network. Returns 1 on success, 0 on failure.
  // NOTE - THIS FUNCTION BLOCKS!
  virtual int write( const char * data, int numBytes );

  //////////
  // Returns a NULL-terminated list of Connections with data ready to be
  // read from them. Caller is responsible for deleting the list.
  virtual Connection ** getReadyToRead();

  //////////
  // Returns whether two connections are equal.
  virtual bool isEqual( const Connection& c);

  
  //////////
  // Closes the connection.
  virtual void close();
  
protected:
  Socket theSocket;                           // The low level socket...
  static list<PTPConnection*> connectionList; // List of PTP connections.
  static CrowdMonitor connectionListLock;     // Lock for list.
  static bool listChanged;                    // Has the list changed?
  static Socket * socketList;                 // List of sockets available.
  static Connection ** getReadyConns();       // Static version of
                                              // getReadyToRead().
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:27  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:46  simpson
// Adding CollabVis files/dirs
//
// Revision 1.7  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.6  2001/07/16 20:29:36  luke
// Updated network stuff...
//
// Revision 1.5  2001/05/30 23:54:58  luke
// Endian driver now can connect to any host
//
// Revision 1.4  2001/05/01 20:55:56  luke
// Works for a single client, but client disconnect causes the server to seg fault
//
// Revision 1.3  2001/04/11 17:47:25  luke
// Net connections and net interface work, but with a few bugs
//
// Revision 1.2  2001/04/05 22:28:01  luke
// Documentation done
//
