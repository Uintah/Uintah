/*
 *
 * Transfer: Message that encapsulates a 'transfer' change.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __TRANSFERM_H_
#define __TRANSFERM_H_

#include <Message/MessageBase.h>
#include <Logging/Log.h>

namespace SemotusVisum {


/**
 * This class provides the infrastructure to create, read, and serialize
 * a Transfer message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Transfer : public MessageBase {
public:

  /**
   *  Constructor.
   *
   */
  Transfer();

  /**
   *  Destructor.
   *
   */
  ~Transfer();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Sets the name of the transfer.
   *
   * @param name   String name of transfer type.
   */
  inline void setTransferType( const string name ) {
    transferType = name;
  }
  
  /**
   * Returns the name of the transfer type.
   *
   * @return String name of transfer type.
   */
  inline string getTransferType() {
    return transferType;
  }
  
  /**
   * Returns true if the server okayed the transfer type.
   *
   * @return True if the server okayed the transfer type.
   */
  inline bool isTransferValid() {
    return transferValid;
  }
  
  /**
   * Returns a Transfer message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static Transfer * mkTransfer( void * data );
  
protected:
  /** True if server responded positively */
  bool   transferValid;

  /** Compressor name */
  string transferType;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:29  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:13  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
