#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


#
#   moveToCursor window
#
#   Author: J. Davison de St. Germain
#
#
#   Moves the given "window" to near the cursor location.  It also
#   withdraws the window.  This is because it needs to determine the
#   size of the window for proper positioning of the window near the
#   edges of the screen.  If you want the window to be remapped, 
#   set the _optional_ parameter to "leave_up":
#
#   moveToCursor $w "leave_up"
#
#   or use the default version
#
#   moveToCursor $w
#

set screenWidth [winfo screenwidth .]
set screenHeight [winfo screenheight .]

proc moveToCursor { window { leave_up "no" } } {

  if { ![envBool SCIRUN_GUI_MoveGuiToMouse] } {
     if { $leave_up == "leave_up" } {
        wm deiconify $window
     }
     return
  }

  # If we are currently running a script... ie, we are loading the net
  # from a file, then do not move GUI to the mouse.
  if [string length [info script]] {
      return
  }

  global screenHeight screenWidth

  set cursorXLoc [expr [winfo pointerx .]]
  set cursorYLoc [expr [winfo pointery .]]

  # After fixing BioPSEFilebox.tcl, I need to at least thank Samsonov
  # because his comments did clue me in on how to get the width and
  # height of a widget.  You have to "withdraw" it first and call
  # "update idletasks" to make it figure out its geometry ... so now
  # this will work!

  if { [winfo ismapped $window] == 0 } {
      if { $leave_up != "leave_up" } {
	  wm withdraw $window
      }
      ::update idletasks
  }

  set guiWidth [winfo reqwidth $window]
  set guiHeight [winfo reqheight $window]

  if { $cursorXLoc < 100 } {
      set windowXLoc [expr $cursorXLoc / 2]
  } elseif { $cursorXLoc > ($screenWidth - $guiWidth) } {
      set windowXLoc [expr $screenWidth - $guiWidth - 20]
  } else {
      set windowXLoc [expr $cursorXLoc - 80]
  }

  if { $cursorYLoc < 100 } {
      set windowYLoc [expr $cursorYLoc / 2]
  } elseif { $cursorYLoc > ($screenHeight - $guiHeight) } {
      set windowYLoc [expr $screenHeight - $guiHeight - 50]
  } else {
      set windowYLoc [expr $cursorYLoc - 80]
  }

  wm geometry $window +$windowXLoc+$windowYLoc

  if { $leave_up == "leave_up" } {
      wm deiconify $window
  }
}
