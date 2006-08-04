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


namespace SemotusVisum {

/**
 * This class provides the infrastructure to create, read, and serialize
 * a Chat message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Chat : public MessageBase {
public:

  /**
   *  Constructor. 
   *
   */
  Chat();

  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~Chat();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Returns the name of the client initiating the message.
   *
   * @return  Name of the client initiating the message.
   */
  inline string getName() { return name; }

  /**
   *  Returns the text of the message.
   *
   * @return The text of the message.
   */
  inline string getText() { return text; }

  /**
   *   Sets the name of the client initiating the message.
   *
   * @param theName      The name of the client initiating the message.  
   */
  inline void   setName( const string theName ) {
    name = theName;
  }

  /**
   *  Sets the text of the message.
   *
   * @param theText   The text of the message.    
   */
  inline void setText( const string theText ) {
    text = theText;
  }

  /**
   *  Returns a Chat message from the given raw data.
   *
   * @param data   Raw data
   * @return New message, or NULL on error.
   */
  static Chat * mkChat( void * data );
  
protected:
  /** Client name initiating the chat message. */
  string   name;

  /* Text of the chat message. */
  string   text;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:24  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:08  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/31 22:52:05  luke
// Added chat message
//
