/*
 *
 * Collaborate: Message that encapsulates a collaborate message.
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

#ifndef __COLLABORATE_H_
#define __COLLABORATE_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>


namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Collaborate
   
KEYWORDS
   Collaborate, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Collaborate message.
   
****************************************/
class Collaborate : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are incoming.
  Collaborate( bool request = false );

  //////////
  // Destructor. Deallocates all memory.
  ~Collaborate();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Returns the text of the message.
  inline char * getText() { return text; }

  //////////
  // Sets the text of the message.
  inline void setText( const char * theText ) {
    if ( !theText ) return;
    if ( text ) delete text;
    text = strdup( theText );
  }

  //////////
  // Changes any local annotation IDs to the new ID specified.
  void switchID( const char * newID );
  
  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }

  //////////
  // Returns a Collaborate message from the given raw data.
  static Collaborate * mkCollaborate( void * data );
  
protected:

  // True if this message is a request.
  bool     request;

  // Text of the collaborate message.
  char *   text;

  // Returns the index of the next "local" string in the text, or -1 if
  // no more exist.
  int nextLocal( int start, const char * text );
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
// Revision 1.2  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.1  2001/09/23 02:24:11  luke
// Added collaborate message
//

