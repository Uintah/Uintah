/*
 *
 * MessageBase: Base class for all messages.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __MESSAGE_BASE_H_
#define __MESSAGE_BASE_H_

#include <stdio.h>
#include <XML/XML.h>
#include <Malloc/Allocator.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

const char * const dataMarker = "\001\002\003\004\005";

/**************************************
 
CLASS
   MessageBase
   
KEYWORDS
   Message
   
DESCRIPTION

   This class provides a few basic services to the MessageBase subclasses.
   
****************************************/
class MessageBase {
public:

  //////////
  // Constructor. Sets the serial output to NULL, initializes XML.
  MessageBase() : output( NULL ), finished( false ) {
    XMLI::initialize();
  }

  //////////
  // Destructor.
  virtual ~MessageBase() { if ( output ) delete output; }

  //////////
  // Finishes serializing the message.
  virtual void           finish() = 0;

  //////////
  // Returns the serialized output.
  virtual inline  char * getOutput() { return output; }
  
protected:
  virtual inline  void   mkOutput( char * out ) {
    output =
      scinew char[ strlen( out ) + strlen( dataMarker ) + 1];
    sprintf( output, "%s%s", out, dataMarker );
  }
  
  // Serialized message output.
  char * output;

  // If we have already serialized our output.
  bool finished;
};



}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:19  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:04  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.5  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.4  2001/05/21 19:19:29  luke
// Added data marker to end of message text output
//
// Revision 1.3  2001/05/14 22:39:01  luke
// Fixed compiler warnings
//
// Revision 1.2  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.1  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
