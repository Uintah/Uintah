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
#  GenClock.tcl
#
#  Written by:
#   Allen Sanderson
#   SCI Institute
#   University of Utah
#   January 2004
#
#  Copyright (C) 2004 SCI Institute
#
itcl_class SCIRun_Visualization_GenClock {
    inherit Module
   
    constructor {config} {
        set name genClock
        set_defaults
    }

    method set_defaults {} {
	global $this-type
	global $this-showtime
	global $this-bbox
	global $this-format
	global $this-min
	global $this-max
	global $this-current
	global $this-size
	global $this-location
	global $this-color-r
	global $this-color-g
	global $this-color-b

	set $this-type 0
	set $this-showtime 0
	set $this-bbox 1
	set $this-format "%8.3f seconds"
	set $this-min 0
	set $this-max 1
	set $this-current 0
	set $this- 1.0
	set $this-size 100
	set $this-location "Top Left"
	set $this-color-r 1.0
	set $this-color-g 1.0
	set $this-color-b 1.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
	wm title $w "Gen Clock"
	
# Type
	iwidgets::labeledframe $w.type -labeltext "Clock Type"
	set type [$w.type childsite]

# Type - digital
	frame $type.digital

	radiobutton $type.digital.button -variable $this-type -value 1 \
	     -command "$this-c needexecute"
	label $type.digital.label -text "Digital" -width 8 \
	    -anchor w -just left
	
	pack $type.digital.button $type.digital.label -side left

# Type - analog
	frame $type.analog

	radiobutton $type.analog.button -variable $this-type -value 0 \
	     -command "$this-c needexecute"
	label $type.analog.label -text "Analog" -width 8 \
	    -anchor w -just left
	
	pack $type.analog.button $type.analog.label -side left

	label $type.timelabel -text "Time" -width 5 \
	    -anchor w -just left
	

# Type - analog - time
	frame $type.time

# Type - analog - time - hide
	frame $type.time.hide

	radiobutton $type.time.hide.button -variable $this-showtime -value 0 \
	     -command "$this-c needexecute"
	label $type.time.hide.label -text "Hide" -width 6 \
	    -anchor w -just left
	
	pack $type.time.hide.button $type.time.hide.label -side left

# Type - analog - hide - show
	frame $type.time.show

	radiobutton $type.time.show.button -variable $this-showtime -value 1 \
	     -command "$this-c needexecute"
	label $type.time.show.label -text "Show" -width 6 \
	    -anchor w -just left
	
	pack $type.time.show.button $type.time.show.label -side left


	pack $type.time.show $type.time.hide -side top

	pack $type.digital $type.analog $type.timelabel $type.time -side left
	pack $w.type -fill x -expand yes -side top


# Style
	iwidgets::labeledframe $w.style -labeltext "Clock Style"
	set style [$w.style childsite]

# Style - box
	frame $style.bbox

	checkbutton $style.bbox.button -variable $this-bbox \
	     -command "$this-c needexecute"
	label $style.bbox.label -text "Box" -width 4 \
	    -anchor w -just left
	
	pack $style.bbox.button $style.bbox.label -side left

	pack $style.bbox -side left

# Style - color
	frame $style.color
	addColorSelection $style.color "Color" $this-color "color_change"
	pack $style.color -side left -padx 5

	pack $w.style -fill x -expand yes -side top

# Style - format
	frame $style.format
	label $style.format.label -text "C Style Format" -width 15 \
	    -anchor w -just left
	entry $style.format.entry -width 16 -text $this-format

	pack $style.format.label $style.format.entry -side left

	pack $style.format -side left -padx 5


# Range
	iwidgets::labeledframe $w.range -labeltext "Analog Clock Range"
	set range [$w.range childsite]

# Range - minimum
	frame $range.min
	label $range.min.label -text "Min."  -width 5 -anchor w -just left
	entry $range.min.entry -width 6 -text $this-min

	pack $range.min.label $range.min.entry -side left
	pack $range.min -side left

# Range - maximum
	frame $range.max
	label $range.max.label -text "Max."  -width 5 -anchor w -just left
	entry $range.max.entry -width 6 -text $this-max

	pack $range.max.label $range.max.entry -side left
	pack $range.max -side left -padx 5

# Range - current
	frame $range.current
	label $range.current.label -text "Current"  -width 7 -anchor w -just left
	entry $range.current.entry -width 6 -text $this-current

	pack $range.current.label $range.current.entry -side left
	pack $range.current -side left -padx 5

	pack $w.range -fill x -expand yes -side top

# Size
	iwidgets::labeledframe $w.size -labeltext "Clock Size"
	set size [$w.size childsite]

# Size - small
	frame $size.small

	radiobutton $size.small.button -variable $this-size -value 50 \
	    -command "$this-c needexecute"
	label $size.small.label -text "Small" -width 6 \
	    -anchor w -just left
	
	pack $size.small.button $size.small.label -side left
	pack $size.small -side left -padx 5

# Size - medium
	frame $size.medium

	radiobutton $size.medium.button -variable $this-size -value 100 \
	    -command "$this-c needexecute"
	label $size.medium.label -text "Medium" -width 6 \
	    -anchor w -just left
	
	pack $size.medium.button $size.medium.label -side left
	pack $size.medium -side left -padx 5

# Size - large
	frame $size.large

	radiobutton $size.large.button -variable $this-size -value 150 \
	    -command "$this-c needexecute"
	label $size.large.label -text "Large" -width 6 \
	    -anchor w -just left
	
	pack $size.large.button $size.large.label -side left
	pack $size.large -side left -padx 5

# Size - custom
	frame $size.custom
	label $size.custom.label -text "Custom"  -width 7 -anchor w -just left
	entry $size.custom.entry -width 4 -text $this-size
	label $size.custom.percent -text "%"  -width 1 -anchor w -just left

	pack $size.custom.label $size.custom.entry $size.custom.percent \
	    -side left
	pack $size.custom -side left -padx 5
	
	pack $w.size -fill x -expand yes -side top



# Location
	iwidgets::labeledframe $w.location -labeltext "Clock Location"
	set location [$w.location childsite]

# Location - top left
	frame $location.top_left

	radiobutton $location.top_left.button -variable $this-location \
	    -value "Top Left" -command "$this-c needexecute"
	label $location.top_left.label -text "Top Left" -width 9 \
	    -anchor w -just left
	
	pack $location.top_left.button $location.top_left.label -side left

# Location - top right
	frame $location.top_right

	radiobutton $location.top_right.button -variable $this-location \
	    -value "Top Right" -command "$this-c needexecute"
	label $location.top_right.label -text "Top Right" -width 10 \
	    -anchor w -just left
	
	pack $location.top_right.button $location.top_right.label -side left

# Location - bottom left
	frame $location.bottom_left

	radiobutton $location.bottom_left.button -variable $this-location \
	    -value "Bottom Left" -command "$this-c needexecute"
	label $location.bottom_left.label -text "Bottom Left" -width 12 \
	    -anchor w -just left
	
	pack $location.bottom_left.button $location.bottom_left.label -side left

# Location - bottom right
	frame $location.bottom_right

	radiobutton $location.bottom_right.button -variable $this-location \
	    -value "Bottom Right" -command "$this-c needexecute"
	label $location.bottom_right.label -text "Bottom Right" -width 13 \
	    -anchor w -just right
	
	pack $location.bottom_right.button $location.bottom_right.label -side left


	pack $location.top_left $location.top_right \
	    $location.bottom_left $location.bottom_right -side left
	
	pack $w.location -fill x -expand yes -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     SciRaise $window.color
	     return
	 } else {
	     makeColorPicker $window.color $color \
		     "$this setColor $col $color $colMsg" \
		     "destroy $window.color"
	 }
   }

    method setColor {col color colMsg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $colMsg

# The above works for only the geometry not for the text so execute.
	 $this-c needexecute
    }

    method addColorSelection {frame text color colMsg} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $frame.colorFrame
	 frame $frame.colorFrame.col -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
	 button $frame.colorFrame.set_color \
		 -text $text -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left -padx 2
	 pack $frame.colorFrame -side left
    }
}

