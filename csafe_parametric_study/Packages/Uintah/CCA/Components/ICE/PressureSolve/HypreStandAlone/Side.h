/*#############################################################################
  # Side.h - Global macros.
  ###########################################################################*/

#ifndef _SIDE_H
#define _SIDE_H

#include "DebugStream.h"

enum Side {  // Left/right boundary in each dim
  Left = -1,
  Right = 1,
  NA = 3
};

Side& operator++(Side &s);
std::ostream& operator << (std::ostream& os, const Side& s);

#endif // _SIDE_H
