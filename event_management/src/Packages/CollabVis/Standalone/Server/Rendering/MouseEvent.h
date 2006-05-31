/*
 *
 * MouseEvent: General info for a mouse event.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __MouseEvent_h_
#define __MouseEvent_h_

#include <sys/time.h>
#include <unistd.h>

namespace SemotusVisum {
namespace Rendering {

/**************************************
 
CLASS
   MouseEvent

KEYWORDS
   Rendering, Image Streaming, Mouse
   
DESCRIPTION

   This class provides information about a mouse event - X/Y coordinates,
   which buttons were pressed, actions, and the time the event took place.
   
****************************************/
struct MouseEvent {

  //////////
  // Constructor. Sets params to defaults.
  MouseEvent() : x( -1 ), y( -1 ), button( 'U' ), action(-1) {
  }
  
  //////////
  // Constructor. Sets up all data.
  MouseEvent(int x, int y, char button, int action, struct timeval time) :
    x(x), y(y), button(button), action(action), time(time) {}
  int x;
  int y;
  char button;
  int action;
  struct timeval time;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:37  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:35  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/05/21 22:00:46  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.2  2001/04/05 21:15:21  luke
// Added header and log info
//
