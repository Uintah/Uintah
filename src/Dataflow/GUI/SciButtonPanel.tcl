
#
#   makeSciButtonPanel parent close_window this [-no_execute] [{"btn_name" "command "tip"}...]
#
#   Author: J. Davison de St. Germain
#
#   Adds a separator line followed by two buttons (Execute and Close)
#   to the bottom of the "parent" widget.  "close_window" will be 
#   withdrawn when the 'Close' button is pressed.  "This" will be 
#   executed when the 'Execute' button is pressed.
#
#   -no_execute     <- Removes the execute button
#   [{"btn_name" "command" "tip"}...]  <- Optional buttons and tool tip text (triples) that
#                                         will be placed on the panel (On the left side of
#                                         the standard buttons).  Each button needs to be
#                                         self contained as a single arg.  Eg:
#                                         "Doit \"$this doit\" \"This is a tool tip\""
#
#   NOTE: This function also overrides the "close_window"s destruction
#   window decoration and makes the window 'close' instead of being
#   destroyed.
#

proc makeSciButtonPanel { parent close_window this args } {

  set make_exec_btn "true"

  set outside_pad 4

  frame $parent.separator -height 2 -relief sunken -borderwidth 2
  pack  $parent.separator -fill x -pady 5

  frame $parent.btnBox
  pack  $parent.btnBox -anchor e

  set btnId 0
  # Parse args
  for {set arg 0} {$arg < [llength $args] } { incr arg } {

      set argName [lindex $args $arg]

      if { $argName == "-no_execute" } {
	  set make_exec_btn "false"
      } else {
	  # Add button
	  set name    "[lindex $argName 0]"
	  set command "[lindex $argName 1]"
	  set tip     "[lindex $argName 2]"

	  set size [string length $name]
	  if { $size < 10 } { set size 10 }

	  button $parent.btnBox.btn$btnId -width $size -text "$name" -command "$command"
	  pack   $parent.btnBox.btn$btnId -padx $outside_pad -pady $outside_pad -side left
	  if { "$tip" != "" } {
	      Tooltip $parent.btnBox.btn$btnId "$tip"
	  }
	  incr btnId
      }
  }
  

  if { $make_exec_btn == "true" } {
      button $parent.btnBox.execute -width 10 -text "Execute" -command "$this-c needexecute"
      pack   $parent.btnBox.execute -padx $outside_pad -pady $outside_pad -side left
      Tooltip $parent.btnBox.execute "Instructs SCIRun to run this (and any connected) module(s)"
  }

  button $parent.btnBox.close -width 10 -text "Close" -command "wm withdraw $close_window"
  pack   $parent.btnBox.close -padx $outside_pad -pady $outside_pad -side left
  Tooltip $parent.btnBox.close "Hides this GUI"

  # Vertical separator
  frame $parent.btnBox.separator -width 2 -relief sunken -borderwidth 2
  pack  $parent.btnBox.separator -fill y -padx $outside_pad -pady $outside_pad -side left

  button $parent.btnBox.highlight -width 10 -text "Find" -command "fadeinIcon [$this modname] 1 1"
  pack   $parent.btnBox.highlight -padx $outside_pad -pady $outside_pad -side left
  Tooltip $parent.btnBox.highlight "Highlights (on the Network Editor) the\nmodule that corresponds to this GUI"

  # Override the destroy window decoration and make it only close the window
  wm protocol $close_window WM_DELETE_WINDOW "wm withdraw $close_window"
}

