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

#include <Util/stringUtil.h>

#include <Network/Connection.h>
#define ASIP_SHORT_NAMES
#include <Network/AppleSeeds/ipseed.h>

#define ASFMT_SHORT_NAMES
#include <Network/AppleSeeds/formatseed.h>

#include <Thread/CrowdMonitor.h>
namespace SemotusVisum {

using namespace SCIRun;

/**
 * PTPConnection implements a blocking point-to-point connection. It
 * provides safe reads and writes to/from the network.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class PTPConnection : public Connection {
public:

  /**
   * Constructor
   *
   */
  PTPConnection();

  
  /**
   * Constructor that takes in a raw socket.
   *
   * @param theSocket     
   */
  PTPConnection( Socket theSocket );
  
  /**
   *  Constructor that takes a server:port to connect to.
   *
   * @param server        Server name
   * @param port          Port to connect to on server.
   */
  PTPConnection( string server, int port );
  
  /**
   *  Destructor - closes the connection, if open.
   *
   */
  virtual ~PTPConnection();

  /**
   *  Reads the given number of bytes from the network, and stores it
   * into the memory pointed to by data. Returns the number of bytes read.
   * NOTE - THIS FUNCTION BLOCKS!
   *
   * @param data        Pre-allocated space to read data into.
   * @param numBytes    Number of bytes to read
   * @return            Number of bytes actually read.
   */
  virtual int read( char * data, int numBytes );

  /**
   * Writes the given number of bytes from the memory pointed to by data
   * to the network. Returns 1 on success, 0 on failure.
   * NOTE - THIS FUNCTION BLOCKS!
   *
   * @param data       Data to write to the network
   * @param numBytes   Number of bytes to write.
   * @return           1 on success, 0 on failure.
   */
  virtual int write( const char * data, int numBytes );

  /**
   * Is this connection valid?
   *
   * @return  True if the connection is valid (ie, connected)
   */
  virtual bool valid() { return (theSocket != NO_SOCKET); }
  
  /**
   *  Returns whether two connections are equal.
   *
   * @param c     Connection to test.
   * @return      True if the connections are equal.
   */
  virtual bool isEqual( const Connection& c);


  /**
   *  Closes the connection.
   *
   */
  virtual void close();
  
protected:
  /** The raw, low-level socket */
  Socket theSocket;               
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:33  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:01  simpson
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
