
#
#   moveToCursor window
#
#   Author: J. Davison de St. Germain
#
#   Moves the given "window" to near the cursor location.
#

proc moveToCursor { window } {

  set mouseXLoc [expr [winfo pointerx .] - 70]
  set mouseYLoc [expr [winfo pointery .] - 70]

  wm geometry $window +$mouseXLoc+$mouseYLoc
}
