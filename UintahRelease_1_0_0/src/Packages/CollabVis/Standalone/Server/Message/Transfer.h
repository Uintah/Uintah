/*
 *
 * Transfer: Message that encapsulates a 'transfer' - disconnect.
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
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Transfer
   
KEYWORDS
   Transfer, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Transfer message.
   
****************************************/
class Transfer : public MessageBase {
public:

  //////////
  // Constructor. By default, all messages are incoming.
  Transfer( bool request = true );

  //////////
  // Destructor.
  ~Transfer();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Sets the 'okay' parameter in the message, if not a request.
  inline void setOkay( bool okay, const char * transfer ) {
    if ( okay )
      this->okay = 1;
    else
      this->okay = 0;
    if ( transfer )
      this->transfer = strdup( transfer );
  }

  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }

  //////////
  // Returns the name of the transfer mode
  inline char * getName() { return transfer; }
  
  //////////
  // Sets the name of the transfer mode
  inline void   setName( const char * name ) {
    if ( name ) transfer = strdup( name );
  }
  //////////
  // Returns a Transfer message from the given raw data.
  static Transfer * mkTransfer( void * data );
  
protected:

  // 0 if has problems, 1 if okay, -1 if not yet set.
  int    okay;
  
  // True if this message is a request.
  bool   request;

  // Transfer Mode name
  char * transfer;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
