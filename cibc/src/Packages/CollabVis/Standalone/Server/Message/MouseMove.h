/*
 *
 * MouseMove: Message that encapsulates mouse button press.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __MOUSE_MOVE_H_
#define __MOUSE_MOVE_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {


/**************************************
 
CLASS
   MouseMove
   
KEYWORDS
   MouseMove, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a MouseMove message.
   
****************************************/
class MouseMove : public MessageBase {
public:

  ///////////
  // Enumeration for mouse actions.
  enum {
    START,
    DRAG,
    END,
    UNKNOWN
  };
  
  //////////
  // Constructor.
  MouseMove();

  //////////
  // Destructor.
  ~MouseMove();

  //////////
  // Finishes serializing the message.
  void finish();
 
  //////////
  // Sets the new mouse button press params to those given.
  inline void setMove( int x, int y, char button, int action ) {
    this->x = x; this->y = y; this->button = button;
    this->action = action;
  }

  //////////
  // Fills in the given params with the current mouse button press params.
  inline void getMove( int &x, int &y, char &button, int& action ) const {
    x = this->x; y = this->y; button = this->button;
    action = this->action;
  }

  //////////
  // Returns a MouseMove message from the given raw data.
  static MouseMove * mkMouseMove( void * data );
  
protected:
  // Coordinates of the mouse event.
  int x, y;

  // Button pressed.
  char button;

  // Mouse action
  int action;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:04  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
