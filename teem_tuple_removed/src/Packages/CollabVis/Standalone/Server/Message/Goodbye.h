/*
 *
 * Goodbye: Message that encapsulates a 'goodbye' - disconnect.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __GOODBYE_H_
#define __GOODBYE_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Goodbye
   
KEYWORDS
   Goodbye, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Goodbye message.
   
****************************************/
class Goodbye : public MessageBase {
public:

  //////////
  // Constructor. By default, all messages are incoming.
  Goodbye( bool request = true );

  //////////
  // Destructor.
  ~Goodbye();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }
  
  //////////
  // Returns a Goodbye message from the given raw data.
  static Goodbye * mkGoodbye( void * data );
  
protected:
  
  // True if this message is a request.
  bool   request;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
