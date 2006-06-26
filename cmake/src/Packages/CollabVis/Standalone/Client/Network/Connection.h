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


/**
 * Connection is the superclass of different types of network connections.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Connection {
public:

  /**
   *  Constructor
   *
   */
  Connection() {}

  /**
   *  Destructor
   *
   */
  virtual ~Connection() {}

  /**
   *  Prototype for reading data from the network.
   *
   * @param data       Preallocated space for data
   * @param numBytes   Number of bytes to read
   * @return  Varies
   */
  virtual int read ( char * data, int numBytes ) = 0;

  /**
   *  Prototype for writing data to the network.
   *
   * @param data       Data to write
   * @param numBytes   Number of bytes to write
   * @return           Varies
   */
  virtual int write( const char * data, int numBytes ) = 0;

  /**
   *  Is this connection valid?
   *
   * @return True if valid; else false
   */
  virtual bool valid() = 0;
  
  /**
   * Returns whether two connections are equal.
   *
   * @param c     Connection to compare
   * @return      True if equal; else false.
   */
  virtual bool isEqual( const Connection& c) = 0;

  /**
   * Closes the connection.
   *
   */
  virtual void close() = 0;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:31  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:59  simpson
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
