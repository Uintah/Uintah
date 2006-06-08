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

namespace SemotusVisum {


/** Enumeration for mouse actions. */
typedef enum {
  START, /** Start motion */
  DRAG,  /** Dragging */
  END,   /** End motion */
  UNKNOWN /** Unknown action */
} action_t;


/**
 * This class provides the infrastructure to create, read, and serialize
 * a MouseMove message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class MouseMove : public MessageBase {
public:
  
  /**
   *  Constructor.
   *
   */
  MouseMove();

  /**
   *  Destructor.
   *
   */
  ~MouseMove();

  /**
   *   Finishes serializing the message.
   *
   */
  void finish();
 
  /**
   *  Sets the new mouse button press params to those given.
   *
   * @param x       X coordinate
   * @param y       Y coordinate
   * @param button  Which button      
   * @param action  Mouse action
   */
  inline void setMove( const int x, const int y, const char button,
		       const action_t action ) {
    this->x = x; this->y = y; this->button = button;
    this->action = action;
  }

  
  /**
   *  Fills in the given params with the current mouse button press params.
   *
   * @param x       X coordinate
   * @param y       Y coordinate
   * @param button  Which button      
   * @param action  Mouse action
   */
  inline void getMove( int &x, int &y, char &button, action_t& action ) const {
    x = this->x; y = this->y; button = this->button;
    action = this->action;
  }

protected:
  /** Coordinates of the mouse event. */
  int x, y;

  /** Button pressed. */
  char button;

  /** Mouse action */
  action_t action;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:28  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:12  simpson
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
