
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

  global screenHeight screenWidth

  set cursorXLoc [expr [winfo pointerx .]]
  set cursorYLoc [expr [winfo pointery .]]

  # Neither commands "winfo width $window", or "winfo reqwidth
  # $window" return the windows (soon to be) width.  (Ie: the window
  # has not yet been "realized", and has a width of '1'.  Therefore I
  # am just goin to assume GUI window size of 300x300 for now.
  # This works fairly well for all but really big windows.
  set guiWidth 300
  set guiHeight 300

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
      set windowYLoc [expr $screenHeight - $guiHeight - 20]
  } else {
      set windowYLoc [expr $cursorYLoc - 80]
  }

  wm geometry $window +$windowXLoc+$windowYLoc
}
