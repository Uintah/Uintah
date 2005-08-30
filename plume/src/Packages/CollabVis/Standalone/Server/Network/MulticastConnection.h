/*
 *
 * MulticastConnection: Provides a multicast network connection
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
#include <pthread.h>

#include <string>

namespace SemotusVisum {
namespace Network {

#define MAXIMUM_DATA_SIZE        1507328 // maximum total number of bytes to send

/**************************************
 
CLASS
   MulticastConnection
   
KEYWORDS
   Network, Connection, Multicast
   
DESCRIPTION

   MulticastConnection implements a blocking multicast connection. It
   provides safe writes to the network.
   
****************************************/
class MulticastConnection : public Connection {
public:

  //////////
  // Constructor
  MulticastConnection();

  //////////
  // Constructor, which provides the multicast group and port to join.
  MulticastConnection( const char * group, const int port );

  //////////
  // Destructor
  virtual ~MulticastConnection();

  //////////
  // The read() function is not implemented.
  virtual int read ( char * data, int numBytes );

  //////////
  // Writes the given number of bytes from the memory pointed to by
  // data to the network. Returns the number of bytes written.
  virtual int write( const char * data, int numBytes );

  //////////
  // Returns a list of Connections with data ready to be read from them.
  // Caller is responsible for deleting the list.
  virtual Connection ** getReadyToRead();

  //////////
  // Returns whether two connections are equal.
  virtual bool isEqual( const Connection& c);

  
  //////////
  // Closes the connection.
  virtual void close();

protected:

  int			sendfd; // The low-level socket.
  socklen_t			salen;
  struct sockaddr		*sasend; // Low-level socket address

};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:24  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:43  simpson
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
