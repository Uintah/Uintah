/*
 *
 * Connection: Superclass for different types of network connections.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2001
 *
 */

#ifndef __Connection_h_
#define __Connection_h_

#define ASIP_SHORT_NAMES
//#include <Network/AppleSeeds/ipseed.h>

#define SV_TRANSFER_MODE 0

namespace SemotusVisum {
namespace Network {

/**************************************
 
CLASS
   Connection
   
KEYWORDS
   Network, Connection
   
DESCRIPTION

   Connection is the superclass of different types of network connections.
   
****************************************/

////////
// Timeout (in seconds) for getReadyToRead
const double Timeout = 1.0;

class Connection {
public:

  //////////
  // Constructor
  Connection() {}

  //////////
  // Destructor
  virtual ~Connection() {}

  //////////
  // Prototype for reading data from the network.
  virtual int read ( char * data, int numBytes ) = 0;

  //////////
  // Prototype for writing data to the network.
  virtual int write( const char * data, int numBytes ) = 0;

  //////////
  // Returns a list of connections with data ready to be read from them.
  virtual Connection ** getReadyToRead() = 0;

  //////////
  // Returns whether two connections are equal.
  virtual bool isEqual( const Connection& c) = 0;

  //////////
  // Closes the connection.
  virtual void close() = 0;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:23  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:42  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/07/16 20:29:36  luke
// Updated network stuff...
//
// Revision 1.3  2001/04/11 17:47:24  luke
// Net connections and net interface work, but with a few bugs
//
// Revision 1.2  2001/04/05 22:28:00  luke
// Documentation done
//
