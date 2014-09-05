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


#  Dial.tcl
#  Written by:
#   James Purciful and Dave Weinstein
#   Department of Computer Science
#   University of Utah
#   May 1995
#  Copyright (C) 1995 SCI Group

# The legal dialtypes are:
# (Use (dialtype)_dial to create a dial.)
# bounded:     between min and max, inclusive
# unbounded:   infinite crank mode
# wrapped:     between min and max, but goes from min to max
#              and vice versa (e.g. 0-360 degree angle)
#              (min IS max)

# min and max must be valid for bounded and wrapped.

# BaseDialW

itcl_class DialW {
    inherit BaseDialW
    
    # None of these are required.  None work after initialization.
    # Use either text or textVariable, not both.
    public initialDialtype ""
    public typeVariable ""
    public dialtypeVariable ""
    public text ""
    public textVariable ""
    public variable ""
    public initialValue ""
    # One revolution is scale units.
    public initialScale ""
    public scaleVariable ""
    public initialMin ""
    public minVariable ""
    public initialMax ""
    public maxVariable ""
    public command ""
    
    constructor {config} {
	BaseDialW::constructor

	global $dialtypeVariable
	if {[info exists $dialtypeVariable] == 0} {
	    global $this-dialtype
	    set typeVariable $this-dialtype
	    if {$initialDialtype == ""} {
		set initialDialtype "unbounded"
	    }
	    set $dialtypeVariable $initialDialtype
	} else {
	    if {$initialDialtype != ""} {
		set $dialtypeVariable $initialDialtype
	    }
	}

	global $variable
	if {[info exists $variable] == 0} {
	    global $this-value
	    set variable $this-value
	    if {$initialValue == ""} {
		set initialValue 0.0
	    }
	    set $variable $initialValue
	} else {
	    if {$initialValue != ""} {
		set $variable $initialValue
	    } else {
		set initialValue [set $variable]
	    }
	}
	
	global $scaleVariable
	if {[info exists $scaleVariable] == 0} {
	    global $this-scale
	    set scaleVariable $this-scale
	    if {$initialScale == ""} {
		set initialScale 1.0
	    }
	    set $scaleVariable $initialScale
	} else {
	    if {$initialScale != ""} {
		set $scaleVariable $initialScale
	    }
	}

	global $minVariable
	if {[info exists $minVariable] == 0} {
	    global $this-min
	    set minVariable $this-min
	    if {$initialMin == ""} {
		set initialMin 0.0
	    }
	    set $minVariable $initialMin
	} else {
	    if {$initialMin != ""} {
		set $minVariable $initialMin
	    }
	}

	global $maxVariable
	if {[info exists $maxVariable] == 0} {
	    global $this-max
	    set maxVariable $this-max
	    if {$initialMax == ""} {
		set initialMax 100.0
	    }
	    set $maxVariable $initialMax
	} else {
	    if {$initialMax != ""} {
		set $maxVariable $initialMax
	    }
	}

	label $this.ui.value -textvariable $variable
	global $textVariable
	if {[info exists $textVariable] == 0} {
	    label $this.ui.label -text "$text"
	} else {
	    label $this.ui.label -textvariable $textVariable
	}
	pack $this.ui.value -in $this.ui -side top -fill both
	pack $this.ui.label -in $this.ui -side top -fill both
	
	$c bind ${basetag}Dial <ButtonRelease-3> "$this b3DialRelease"
	$c bind ${basetag}Dial <1> "$this b1DialDown %x %y"
	$c bind ${basetag}Dial <B1-Motion> "$this b1DialMove %x %y" 
	$c bind ${basetag}Dial <ButtonRelease-1> \
		"$this b1DialRelease %x %y"
    }
    
    method resetPos {} {
	global $variable
	global $scaleVariable
	setPos [expr 1.0-fmod(([set $variable]-$initialValue),[set $scaleVariable]) \
		/ [set $scaleVariable]+0.5]
    }

    # B3 in the dial switches modes
    method b3DialRelease {} {
	toggle
    }

    # When B1 is pressed, the dial starts tracking where the user moves 
    # the mouse
    method b1DialDown {x y} {
	startTrackMouse $x $y
    }

    method b1DialMove {x y} {
	set ll [moveTrackMouse $x $y 0]
	if {$mode == "in"} {
	    dialRot [lindex $ll 0] 10
	} else {
	    dialRot [lindex $ll 0] 1
	}
    }

    # B1 release.  Tell the dial not to track motion any more
    method b1DialRelease {x y} {
	endTrackMouse $x $y
    }

    # Dial is in crank mode -- this gets called with the most recent
    # positional change in radians
    method dialRot {rad rate} {
	global $dialtypeVariable
	global $variable
	global $minVariable
	global $maxVariable
	global $scaleVariable

	set f [expr $rad*4.77*$rate*[set $scaleVariable]/30.0]
	set $variable [format %3.3f [expr $f+[set $variable]]]

	global $dialtypeVariable
	global $minVariable
	global $maxVariable
	if [expr ([string compare [set $dialtypeVariable] "bounded"]!=0)] {
	    if [expr [set $variable] < [set $minVariable]] {
		set $variable [set $minVariable]
	    }
	    if [expr [set $variable] > [set $maxVariable]] {
		set $variable [set $maxVariable]
	    }
	} else if [expr ([string compare [set $dialtypeVariable] "wrapped"] == 0)] {
	    if [expr [set $variable] < [set $minVariable]] {
		set $variable [expr [set $variable] + \
			([set $maxVariable]-[set $minVariable])]
	    }
	    if [expr [set $variable] >= [set $maxVariable]] {
		set $variable [expr [set $variable] - \
			([set $maxVariable]-[set $minVariable])]
	    }
	}	
	
	if {$command != ""} {
	    eval $command [set $variable]
	}
    }
}
