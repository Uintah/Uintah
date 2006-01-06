/*
 *
 * MouseEvent: Abstraction for mouse event
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#ifndef __MouseEvent_h_
#define __MouseEvent_h_

#include <Message/MouseMove.h>

namespace SemotusVisum {

/**
 * Abstraction for mouse event
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class MouseEvent {
public:
  
  /**
   * Constructor
   *
   * @param button        Which button 'R'/'M'/'L'
   * @param action        Type of action
   * @param x             Window X of event
   * @param y             Window Y of event
   * @see                 Message::MouseMove
   */
  MouseEvent( const char button, const action_t action,
	      const int x, const int y ) : button(button), action(action),
					   x(x), y(y) { }

  /// Default Constructor
  MouseEvent( ) {}
  
  /**
   *  Destructor
   *
   */
  ~MouseEvent() {}

  /// Button
  char         button;

  /// Mouse action
  action_t action;

  /// X coordinate
  int          x;

  /// Y coordinate
  int          y;
  
  /// Time of event
  int time;
};

}

#endif
