/*
 *
 * Compression: Message that encapsulates a 'compression' change.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __COMPRESSIONM_H_
#define __COMPRESSIONM_H_

#include <Message/MessageBase.h>
#include <Logging/Log.h>

namespace SemotusVisum {


/**
 * This class provides the infrastructure to create, read, and serialize
 * a Compression message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Compression : public MessageBase {
public:

  /**
   *  Constructor.
   *
   */
  Compression();

  /**
   *  Destructor.
   *
   */
  ~Compression();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Sets the name of the compression.
   *
   * @param name   String name of compression type.
   */
  inline void setCompressionType( const string name ) {
    compressionType = name;
  }
  
  /**
   * Returns the name of the compression type.
   *
   * @return String name of compression type.
   */
  inline string getCompressionType() {
    return compressionType;
  }
  
  /**
   * Returns true if the server okayed the compression type.
   *
   * @return True if the server okayed the compression type.
   */
  inline bool isCompressionValid() {
    return compressionValid;
  }
  
  /**
   * Returns a Compression message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static Compression * mkCompression( void * data );
  
protected:
  /** True if server responded positively */
  bool   compressionValid;

  /** Compressor name */
  string compressionType;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:25  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:09  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
