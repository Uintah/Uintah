
#
#   makeSciButtonPanel parent close_window this
#
#   Author: J. Davison de St. Germain
#
#   Adds a separator line followed by two buttons (Execute and Close)
#   to the bottom of the "parent" widget.  "close_window" will be 
#   withdrawn when the 'Close' button is pressed.  "This" will be 
#   executed when the 'Execute' button is pressed.
#
#   NOTE: This function also overrides the "close_window"s destruction
#   window decoration and makes the window 'close' instead of being
#   destroyed.
#

proc makeSciButtonPanel { parent close_window this } {

  set outside_pad 4

  frame $parent.separator -height 2 -relief sunken -borderwidth 2
  pack  $parent.separator -fill x

  frame $parent.btnBox
  pack  $parent.btnBox -anchor e

  button $parent.btnBox.execute -width 10 -text "Execute" -command "$this-c needexecute"
  pack   $parent.btnBox.execute -padx $outside_pad -pady $outside_pad -side left
  Tooltip $parent.btnBox.execute "Instructs SCIRun to run this (and any connected) module(s)"

  button $parent.btnBox.close -width 10 -text "Close" -command "wm withdraw $close_window"
  pack   $parent.btnBox.close -padx $outside_pad -pady $outside_pad -side left
  Tooltip $parent.btnBox.close "Hides this GUI"

  # Vertical separator
  frame $parent.btnBox.separator -width 2 -relief sunken -borderwidth 2
  pack  $parent.btnBox.separator -fill y -padx $outside_pad -pady $outside_pad -side left

  button $parent.btnBox.highlight -width 10 -text "Find" -command "fadeinIcon [$this modname]"
  pack   $parent.btnBox.highlight -padx $outside_pad -pady $outside_pad -side left
  Tooltip $parent.btnBox.highlight "Highlights (on the Network Editor) the\nmodule that corresponds to this GUI"

  # Override the destroy window decoration and make it only close the window
  wm protocol $close_window WM_DELETE_WINDOW "wm withdraw $close_window"
}

