
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
  global env

  if { [info exists env(SCI_GUI_MoveGuiToMouse)] &&
       ![boolToInt $env(SCIRUN_GUI_MoveGuiToMouse)] } return

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
