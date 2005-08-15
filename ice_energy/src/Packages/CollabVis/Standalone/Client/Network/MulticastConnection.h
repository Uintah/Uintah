/*
 *
 * MulticastConnection: Provides a multicast network connection.
 * This class is totally broken -- don't use it!
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#ifndef __MulticastConnection_h_
#define __MulticastConnection_h_

#include <Network/Connection.h>
#include <netinet/in.h>

namespace SemotusVisum {

//#define MAXIMUM_DATA_SIZE      65536
#define MAXIMUM_DATA_SIZE        1507328 // maximum total number of bytes to receive

/**
 * MulticastConnection implements a blocking multicast connection. It
 * provides safe writes to the network.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class MulticastConnection : public Connection {
public:

  /**
   * Constructor
   *
   */
  MulticastConnection();

  /**
   * Constructor, which provides the multicast group and port to join.
   *
   * @param group     Group to join
   * @param port      Port to connect to
   */
  MulticastConnection( const char * group, const int port );

  /**
   *  Destructor - closes the connection, if open.
   *
   */
  virtual ~MulticastConnection();

  /**
   *  Reads the given number of bytes from the network, and stores it
   * into the memory pointed to by data. Returns the number of bytes read.
   * NOTE - THIS FUNCTION BLOCKS!
   *
   * @param data        Pre-allocated space to read data into.
   * @param numBytes    Number of bytes to read
   * @return            Number of bytes actually read.
   */
  /* FIXME - old code not yet done! */
  virtual int read ( char * data, int numBytes );

   /**
   *  Implements read function
   *
   * @param data        Pre-allocated space to read data into.
   * @param numBytes    Number of bytes to read
   * @return            Number of bytes actually read.
   */
  virtual int receiveData( char * data, int numBytes );

  /**
   * Writes the given number of bytes from the memory pointed to by data
   * to the network. Returns the number of bytes written.
   * NOTE - THIS FUNCTION BLOCKS!
   *
   * @param data       Data to write to the network
   * @param numBytes   Number of bytes to write.
   * @return           Number of bytes written.
   */
  virtual int write( const char * data, int numBytes );

  /**
   * Is this connection valid?
   *
   * @return  True if the connection is valid (ie, readable/writable)
   */
  virtual bool valid() { return (recvfd != -1); }
  
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
  /// Low-level socket address
  //struct sockaddr_in    addr;

  /// The low-level socket.
  //int                   sock;


  int recvfd;
  //const int on = 1;
  socklen_t salen;
  struct sockaddr *sarecv;
 
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:31  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:00  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/07/16 20:29:36  luke
// Updated network stuff...
//
// Revision 1.4  2001/06/05 17:44:56  luke
// Multicast basics working
//
// Revision 1.3  2001/04/11 17:47:24  luke
// Net connections and net interface work, but with a few bugs
//
// Revision 1.2  2001/04/05 22:28:00  luke
// Documentation done
//
