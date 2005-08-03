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
  set outside_pad 4
  frame $parent.buttonPanel -bd 0
  pack $parent.buttonPanel -fill x
  set parent $parent.buttonPanel

  frame $parent.separator -height 2 -relief sunken -borderwidth 2
  pack  $parent.separator -fill x -pady 5

  frame $parent.btnBox
  pack  $parent.btnBox -anchor e

  # Parse options
  set make_help_btn 1
  set make_exec_btn 1
  set make_close_btn 1
  set make_find_btn 1

  foreach argName $args {
      if { $argName == "-no_help" } {
	  listFindAndRemove args $argName
	  set make_help_btn 0

      } elseif { $argName == "-no_execute" } {
	  listFindAndRemove args $argName
	  set make_exec_btn 0

      } elseif { $argName == "-no_close" } {
	  listFindAndRemove args $argName
	  set make_close_btn 0

      } elseif { $argName == "-no_find" } {
	  listFindAndRemove args $argName
	  set make_find_btn 0
      } 
  }

  if { $make_help_btn } {
      button $parent.btnBox.help -text " ? " \
	  -command "moduleHelp [$this modname]"
      set fnt [eval font create [font actual [$parent.btnBox.help cget -font]]]
      set size [expr [font configure $fnt -size]+4]
      font configure $fnt -size $size -weight bold
      $parent.btnBox.help configure -font $fnt
      pack $parent.btnBox.help -padx $outside_pad -pady $outside_pad -side left

      # Vertical separator
      frame $parent.btnBox.separator2 -width 2 -relief sunken -borderwidth 2
      pack  $parent.btnBox.separator2 -fill y -padx $outside_pad -pady $outside_pad -side left

      # Fast Tooltip
      global tooltipDelayMS
      set backup $tooltipDelayMS
      set tooltipDelayMS 100
      Tooltip $parent.btnBox.help "Open Help Browser"
      set tooltipDelayMS $backup
  }


  set btnId 0
  foreach argName $args {
      # Add button
      set name    [lindex $argName 0]
      set command [lindex $argName 1]
      set tip     [lindex $argName 2]
      
      set size [string length $name]
      if { $size < 10 } { set size 10 }
      
      button $parent.btnBox.btn$btnId -width $size \
	  -text $name -command $command
      pack $parent.btnBox.btn$btnId \
	  -padx $outside_pad -pady $outside_pad -side left

      if { "$tip" != "" } {
	  Tooltip $parent.btnBox.btn$btnId $tip
      }
      incr btnId
  }
  

  if { $make_exec_btn } {
      button $parent.btnBox.execute -width 10 -text "Execute" \
	  -command "$this-c needexecute"
      pack $parent.btnBox.execute \
	  -padx $outside_pad -pady $outside_pad -side left
      Tooltip $parent.btnBox.execute \
	  "Instructs SCIRun to run this (and any connected) module(s)"
  }

  if { $make_close_btn } {
      button $parent.btnBox.close -width 10 -text "Close" \
	  -command "wm withdraw $close_window"
      pack   $parent.btnBox.close -padx $outside_pad -pady $outside_pad -side left
      Tooltip $parent.btnBox.close "Hides this GUI"
      bind $close_window <Escape> "wm withdraw $close_window"
  }

  # Vertical separator
  if { $make_find_btn } {
      frame $parent.btnBox.separator -width 2 -relief sunken -borderwidth 2
      pack  $parent.btnBox.separator \
	  -fill y -padx $outside_pad -pady $outside_pad -side left

      button $parent.btnBox.highlight \
	  -width 10 -text "Find" -command "fadeinIcon [$this modname] 1 1"
      pack $parent.btnBox.highlight \
	  -padx $outside_pad -pady $outside_pad -side left
      Tooltip $parent.btnBox.highlight \
	  "Highlights (on the Network Editor) the\nmodule that corresponds to this GUI"
  }

  # Override the destroy window decoration and make it only close the window
  wm protocol $close_window WM_DELETE_WINDOW "wm withdraw $close_window"
  return $parent
}

