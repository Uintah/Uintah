
#
#   moveToCursor window
#
#   Author: J. Davison de St. Germain
#
#   Moves the given "window" to near the cursor location.
#

set screenWidth [winfo screenwidth .]
set screenHeight [winfo screenheight .]

proc moveToCursor { window } {

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
      wm withdraw $window
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
}
