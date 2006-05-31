/*
 *
 * dataItem: Compound object with data and a size count
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#ifndef __dataItem_h_
#define __dataItem_h_

#include <Message/MessageBase.h>

namespace SemotusVisum {



/**
 * dataItem is a compound object that stores a pointer to arbitrary data,
 * as well as a counter of the number of bytes of data. It also has an
 * optional symbolic type for the data. Finally, it contains a 'copy'
 * parameter that allows the item to clean up its data buffer if needs be.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
struct dataItem {

  /**
   * Default constructor. Shouldn't be called, but is here for convenience. 
   *
   */
  dataItem() : data(NULL), numBytes(-1), copy( false ), type(-1)  {}

  
  /**
   *  Constructor initializes data and data size. 
   *
   * @param data  
   * @param numBytes      
   * @param copy  
   * @param type  
   */
  dataItem(const char * data, int numBytes, bool copy, int type=-1 ) :
    numBytes(numBytes), copy(copy), type( type )  {
      
    this->data = const_cast<char *>(data); 
  }

  /**
   * Copy constructor. Just does a bitwise copy of the data.
   *
   * @param di    
   */
  dataItem( const dataItem &di) :
    data( di.data ), numBytes( di.numBytes ), copy( di.copy ),
    type( di.type ) {
  }

  /**
   * Destructor. Does not deallocate memory!
   *
   */
  ~dataItem() {
  }
  
  /**
   *  Assignment operator. Just does a bitwise copy of the data.
   *
   */
  dataItem operator=(const dataItem &di) {
    data = di.data;
    numBytes = di.numBytes;
    copy = di.copy;
    type = di.type;
    
    return *this;
  }

  /**
   *  Returns the size of the data in bytes.
   *
   * @return Data size
   */
  inline int getSize() const { return numBytes; }

  /**
   * Sets the size of the data.
   *
   * @param size  New data size
   */
  inline void setSize( const int size ) { numBytes = size; }
  
  /**
   * Returns a constant pointer to the data.
   *
   * @return Constant pointer to data.
   */
  inline const char * getData() const { return data; }

  /**
   * Returns true if the data is a copy that should be purged.
   *
   * @return True if the data is a copy that can be deleted.
   */
  inline bool getCopy() const { return copy; }
  
  /**
   *  Returns the type of the data.
   *
   * @return Symbolic type of the data
   */
  inline int getType() const { return type; }

  /**
   *  Deallocates memory if this is a copy.
   *
   */
  inline void purge() {
    if ( copy ) {
      delete data;
      data = NULL;
    }
  }
  
protected:
  /// Actual raw data
  char * data;

  /// Number of bytes of data.
  int    numBytes;

  /// True if this data is a copy.
  bool  copy;

  /// Symbolic type of the data.
  int    type;             

};

class MessageBase;

/**
 * MessageData encapsulates the message generated from the network. It
 * contains both the message and any raw data.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
struct MessageData {

  /**
   * Constructor. Sets all values to NULL.
   *
   */
  MessageData() : message( NULL ), data(NULL), size(-1) {}

  /**
   *  Copy constructor - just copies the pointers, and does no realloc.
   *
   * @param md    
   */
  MessageData( const MessageData &md ) {
    message = md.message;
    data = md.data;
    size = md.size;
  }

  /**
   * Initialization constructor.
   *
   * @param message       Message
   * @param data          Raw data, if any
   * @param size          Number of bytes of raw data
   */
  MessageData( MessageBase * message,
	       char * data=NULL,
	       const int size=-1 ) : message( message ),
				     data(data),
				     size( size ) {}


  /**
   *  Destructor
   *
   */
  ~MessageData() {
    delete message;
  }

  /// Pointer to message.
  MessageBase * message;

  /// Raw data
  char * data;

  /// Number of bytes of data
  int size;
};

}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:33  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:59  simpson
// Adding CollabVis files/dirs
//
// Revision 1.8  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.7  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.6  2001/05/31 17:32:46  luke
// Most functions switched to use network byte order. Still need to alter Image Renderer, and some drivers do not fully work
//
// Revision 1.5  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.4  2001/05/14 22:39:02  luke
// Fixed compiler warnings
//
// Revision 1.3  2001/05/14 18:07:56  luke
// Finished documentation
//
// Revision 1.2  2001/05/12 03:29:11  luke
// Now uses messages instead of XML. Also moved drivers to new location
//
// Revision 1.1  2001/04/11 17:47:25  luke
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
