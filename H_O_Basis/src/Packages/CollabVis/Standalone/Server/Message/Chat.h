/*
 *
 * Chat: Message that encapsulates a chat message.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __CHAT_H_
#define __CHAT_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>


namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Chat
   
KEYWORDS
   Chat, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Chat message.
   
****************************************/
class Chat : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are incoming.
  Chat( bool request = false );

  //////////
  // Destructor. Deallocates all memory.
  ~Chat();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Returns the name of the client initiating the message.
  inline char * getName() { return name; }

  //////////
  // Returns the text of the message.
  inline char * getText() { return text; }

  //////////
  // Sets the name of the client initiating the message.
  inline void   setName( const char * theName ) {
    if ( !theName ) return;
    if ( name ) delete name;
    name = strdup( theName );
  }

  //////////
  // Sets the text of the message.
  inline void setText( const char * theText ) {
    if ( !theText ) return;
    if ( text ) delete text;
    text = strdup( theText );
  }
  
  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }

  //////////
  // Returns a Chat message from the given raw data.
  static Chat * mkChat( void * data );
  
protected:

  // True if this message is a request.
  bool     request;

  // Client name initiating the chat message.
  char *   name;

  // Text of the chat message.
  char *   text;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:17  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:01  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/31 22:52:05  luke
// Added chat message
//
