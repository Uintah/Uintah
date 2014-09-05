#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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
#  CreateViewerCaption.tcl
#
#  Written by:
#   Allen Sanderson
#   SCI Institute
#   University of Utah
#   January 2004
#
#  Copyright (C) 2004 SCI Institute
#
itcl_class SCIRun_Visualization_CreateViewerCaption {
    inherit Module
   
    constructor {config} {
        set name CreateViewerCaption
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
	wm title $w "Gen Title"

# Value
	iwidgets::labeledframe $w.value -labeltext "Title Value"
	set value [$w.value childsite]

# Value - value
	frame $value.value
	label $value.value.label -text "Value"  -width 6 -anchor w -just left
	entry $value.value.entry -width 6 -text $this-value

	pack $value.value.label $value.value.entry -side left


# Value - hide
	frame $value.hide

	radiobutton $value.hide.button -variable $this-showValue -value 0 \
	     -command "$this-c needexecute"
	label $value.hide.label -text "Hide" -width 6 \
	    -anchor w -just left
	
	pack $value.hide.button $value.hide.label -side left

# Value - show
	frame $value.show

	radiobutton $value.show.button -variable $this-showValue -value 1 \
	     -command "$this-c needexecute"
	label $value.show.label -text "Show" -width 6 \
	    -anchor w -just left
	
	pack $value.show.button $value.show.label -side left


	pack $value.value $value.show $value.hide -side left
	pack $w.value -fill x -expand yes -side top

# Style
	iwidgets::labeledframe $w.style -labeltext "Title Style"
	set style [$w.style childsite]

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


	frame $w.twocol
	
# Size
	iwidgets::labeledframe $w.twocol.size -labeltext "Clock Size"
	set size [$w.twocol.size childsite]

# Size - small
	frame $size.small

	radiobutton $size.small.button -variable $this-size -value 50 \
	    -command "$this-c needexecute"
	label $size.small.label -text "Small" -width 6 \
	    -anchor w -just left
	
	pack $size.small.button $size.small.label -side left
	pack $size.small -side top -padx 5

# Size - medium
	frame $size.medium

	radiobutton $size.medium.button -variable $this-size -value 100 \
	    -command "$this-c needexecute"
	label $size.medium.label -text "Medium" -width 6 \
	    -anchor w -just left
	
	pack $size.medium.button $size.medium.label -side left
	pack $size.medium -side top -padx 5

# Size - large
	frame $size.large

	radiobutton $size.large.button -variable $this-size -value 150 \
	    -command "$this-c needexecute"
	label $size.large.label -text "Large" -width 6 \
	    -anchor w -just left
	
	pack $size.large.button $size.large.label -side left
	pack $size.large -side top -padx 5

# Size - custom
	frame $size.custom
	label $size.custom.label -text "Custom"  -width 7 -anchor w -just left
	entry $size.custom.entry -width 4 -text $this-size
	label $size.custom.percent -text "%"  -width 1 -anchor w -just left

	pack $size.custom.label $size.custom.entry $size.custom.percent \
	    -side left
	pack $size.custom -side top -padx 5

# Location
	iwidgets::labeledframe $w.twocol.location -labeltext "Title Location"
	set location [$w.twocol.location childsite]

	set locator [makeStickyLocator $location.gui \
			 $this-location-x $this-location-y \
			 100 100]

	$locator bind movable <ButtonRelease> "$this-c needexecute"

	pack $location.gui -fill x -expand yes -side top


	pack $w.twocol.size $w.twocol.location -fill both -expand yes -side left

	pack $w.twocol -fill x -expand yes -side top

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
	 pack $frame.colorFrame.set_color $frame.colorFrame.col \
				  -side left -padx 2
	 pack $frame.colorFrame -side left
    }
}

