
#
# proc createSciDialog()
#
# author: J. Davison de St. Germain
#
# Creates a yes/no, ok/cancel, etc, type modal dialog that will return
# the users response.  Return values are 1, 2, 3.  Corresponding to
# buttons created with -buton1, -button2, and -button3 respectively.
#
# Usage:
#
#     -title <title>
#        Title of the window.  If no title is specified, then the
#        title defaults to the type of dialog (ie: Question, Warning, Error)
#     -message <message>
#     -button1 <label>
#     -button2 <label>
#     -button3 <label>
#        Buttons will be packed from left to right based on the order specified.
#        You may still only have up to 3 buttons.  If no buttons are specified
#        an "Ok" button will be created (and will return a value of 1 when pressed).
#     -message_tip <tip>
#        Adds a tool tip to the message text.  Must be specified after -message
#     -button1_tip <tip>
#     -button2_tip <tip>
#     -button3_tip <tip>
#        Adds a tool tip to the specified button.  This option MUST be
#        specified after the -button# option.
#     -question 
#        Use a question icon (DEFAULT)
#     -warning
#        Use a warning icon
#     -error
#        Use a error icon
#     -parent <window>
#        If this option is specified, the window will center itself on the parent.
#        If not, the window will be placed near the mouse cursor's current location.
#     - 
#     - 

# For TESTING
#set DataflowTCL .
#source NetworkEditor.tcl
#source Tooltips.tcl

# The following icons (question_bits, warning_bits, error_bits) come from the TCL
# distribution and are just used directly here so that I don't have to figure out
# where these files exist and then load them from the file.

set question_bits {
#define question_width 17
#define question_height 27
static unsigned char question_bits[] = {
   0xf0, 0x0f, 0x00, 0x58, 0x15, 0x00, 0xac, 0x2a, 0x00, 0x56, 0x55, 0x00,
   0x2b, 0xa8, 0x00, 0x15, 0x50, 0x01, 0x0b, 0xa0, 0x00, 0x05, 0x60, 0x01,
   0x0b, 0xa0, 0x00, 0x05, 0x60, 0x01, 0x0b, 0xb0, 0x00, 0x00, 0x58, 0x01,
   0x00, 0xaf, 0x00, 0x80, 0x55, 0x00, 0xc0, 0x2a, 0x00, 0x40, 0x15, 0x00,
   0xc0, 0x02, 0x00, 0x40, 0x01, 0x00, 0xc0, 0x02, 0x00, 0x40, 0x01, 0x00,
   0xc0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0xc0, 0x02, 0x00,
   0x40, 0x01, 0x00, 0xc0, 0x02, 0x00, 0x00, 0x01, 0x00};
}

set warning_bits {
#define warning_width 6
#define warning_height 19
static unsigned char warning_bits[] = {
   0x0c, 0x16, 0x2b, 0x15, 0x2b, 0x15, 0x2b, 0x16, 0x0a, 0x16, 0x0a, 0x16,
   0x0a, 0x00, 0x00, 0x1e, 0x0a, 0x16, 0x0a};
}

set error_bits {
#define error_width 17
#define error_height 17
static unsigned char error_bits[] = {
   0xf0, 0x0f, 0x00, 0x58, 0x15, 0x00, 0xac, 0x2a, 0x00, 0x16, 0x50, 0x00,
   0x2b, 0xa0, 0x00, 0x55, 0x40, 0x01, 0xa3, 0xc0, 0x00, 0x45, 0x41, 0x01,
   0x83, 0xc2, 0x00, 0x05, 0x45, 0x01, 0x03, 0xca, 0x00, 0x05, 0x74, 0x01,
   0x0a, 0xa8, 0x00, 0x14, 0x58, 0x00, 0xe8, 0x2f, 0x00, 0x50, 0x15, 0x00,
   0xa0, 0x0a, 0x00};
}

proc createSciDialog { args } {

  global question_bits warning_bits error_bits

  set outside_pad 4

  toplevel .sci_dialog

  frame .sci_dialog.msgBox
  frame .sci_dialog.btnBox

  pack .sci_dialog.msgBox -padx $outside_pad -pady $outside_pad -fill x
 
  set buttonSpecified false
  set icon $question_bits
  set title ""
  set ::sci_dialog_result ""

  set minBtnSize 8

  set placeInParent "false"

  # Parse the arguments to make the dialog.
  for {set arg 0} {$arg < [llength $args] } { incr arg } {

      set argName [lindex $args $arg]

      if { $argName == "-title" } {
	  incr arg
	  set title [lindex $args $arg]
      } elseif { $argName == "-message" } {
	  incr arg
	  label .sci_dialog.msgBox.message -text "[lindex $args $arg]" -justify left
	  pack .sci_dialog.msgBox.message -padx $outside_pad -pady $outside_pad -expand true -fill both -side right
      } elseif { $argName == "-button1" } {
	  incr arg
	  set btnName [lindex $args $arg]
	  set size [string length $btnName]
	  if { $size < $minBtnSize } { set size $minBtnSize }
	  button .sci_dialog.btnBox.b1 -text $btnName -width $size -command {set ::sci_dialog_result 1 }
	  pack .sci_dialog.btnBox.b1 -side left -padx $outside_pad -pady $outside_pad -expand 1
	  set buttonSpecified true
      } elseif { $argName == "-button2" } {
	  incr arg
	  set btnName [lindex $args $arg]
	  set size [string length $btnName]
	  if { $size < $minBtnSize } { set size $minBtnSize }
	  button .sci_dialog.btnBox.b2 -text $btnName -width $size -command {set ::sci_dialog_result 2 }
	  pack .sci_dialog.btnBox.b2 -side left -padx $outside_pad -pady $outside_pad -expand 1
	  set buttonSpecified true
      } elseif { $argName == "-button3" } {
	  incr arg
	  set btnName [lindex $args $arg]
	  set size [string length $btnName]
	  if { $size < $minBtnSize } { set size $minBtnSize }
	  button .sci_dialog.btnBox.b3 -text $btnName -width $size -command {set ::sci_dialog_result 3 }
	  pack .sci_dialog.btnBox.b3 -side left -padx $outside_pad -pady $outside_pad -expand 1
	  set buttonSpecified true
      } elseif { $argName == "-parent" } {
	  incr arg
	  set parent [lindex $args $arg]
	  # This is a hack at finding the middle... someone needs to think about it more
	  # a few seconds longer.... probably need to know the size of this window... but
	  # we don't have that yet... so perhaps we need to delay this calculation.
	  set windowXLoc [expr [winfo rootx $parent] + ([winfo width $parent]/5)]
	  set windowYLoc [expr [winfo rooty $parent] + ([winfo height $parent]/5)]
	  set placeInParent "true"
      } elseif { $argName == "-question" } {
	  if { $title == "" } { set title "Question" }
	  set icon $question_bits
      } elseif { $argName == "-warning" } {
	  if { $title == "" } { set title "Warning" }
	  set icon $warning_bits
      } elseif { $argName == "-error" } {
	  if { $title == "" } { set title "Error" }
	  set icon $error_bits
      } elseif { $argName == "-message_tip" } {
	  if { ![winfo exists .sci_dialog.msgBox.message] } {
	      puts "Error with createSciDialog.  Message tip specified before message text."
	      destroy .sci_dialog
	      return -1
	  }
	  incr arg
	  Tooltip .sci_dialog.msgBox.message [lindex $args $arg]
      } elseif { $argName == "-button1_tip" } {
	  if { ![winfo exists .sci_dialog.btnBox.b1] } {
	      puts "Error with createSciDialog.  Button tip specified before button label."
	      destroy .sci_dialog
	      return -1
	  }
	  incr arg
	  Tooltip .sci_dialog.btnBox.b1 [lindex $args $arg]
      } elseif { $argName == "-button2_tip" } {
	  if { ![winfo exists .sci_dialog.btnBox.b2] } {
	      puts "Error with createSciDialog.  Button tip specified before button label."
	      destroy .sci_dialog
	      return -1
	  }
	  incr arg
	  Tooltip .sci_dialog.btnBox.b2 [lindex $args $arg]
      } elseif { $argName == "-button3_tip" } {
	  if { ![winfo exists .sci_dialog.btnBox.b3] } {
	      puts "Error with createSciDialog.  Button tip specified before button label."
	      destroy .sci_dialog
	      return -1
	  }
	  incr arg
	  Tooltip .sci_dialog.btnBox.b3 [lindex $args $arg]
      } else {
	  puts "Error with createSciDialog.  Parameter '$argName' is invalid"
	  destroy .sci_dialog
	  return -1
      }
  }

  frame .sci_dialog.separator -height 2 -relief sunken -borderwidth 2
  pack .sci_dialog.separator -fill x

  pack .sci_dialog.btnBox -anchor e -fill x

  wm title .sci_dialog $title

  if { $buttonSpecified == "false" } {
      button .sci_dialog.btnBox.ok -width $minBtnSize -text "Ok" -command { set ::sci_dialog_result 1 }
      pack .sci_dialog.btnBox.ok -padx $outside_pad -pady $outside_pad
  }

  # Create the Icon
  image create bitmap iconImg -data $icon
  label .sci_dialog.msgBox.icon -image iconImg
  pack .sci_dialog.msgBox.icon -side left -padx 10 -pady $outside_pad


  if { $placeInParent == "false" } {
      # Move window to mouse location
      moveToCursor .sci_dialog
  } else {
      # Place the window 
      wm geometry .sci_dialog +$windowXLoc+$windowYLoc
  }

  # Don't let it be resized.
  wm resizable .sci_dialog 0 0

  # Wait for a response
  wm maxsize .sci_dialog 600 2000
  wm deiconify .sci_dialog
  #raise .sci_dialog

  grab .sci_dialog

  # Wait until a button is pressed.
  tkwait variable ::sci_dialog_result

  destroy .sci_dialog

  return $::sci_dialog_result
}

# Test runs:
#
#button .b -text hello
#pack .b
#
#createSciDialog  -title "This is a title" -message "this is a message" -message_tip "this is a tip"
#createSciDialog -title "Hello" -warning -button1 "Hello" -button2 "Goodbye" -button3 "Whats up?" -message "this is a message" -button3_tip "This is the b3 tip" 
#createSciDialog -error -button1 "Hello" -button2 "Goodbye" -button3 "Whats up?" -message "this is an error"

#createSciDialog -error -button1 "Hello" -button2 "Goodbye" -button3 "Whats up?" 

