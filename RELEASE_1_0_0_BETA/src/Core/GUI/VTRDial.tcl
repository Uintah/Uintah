#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

#  VTRDial.tcl
#  Written by:
#   James Purciful and Dave Weinstein
#   Department of Computer Science
#   University of Utah
#   May 1995
#  Copyright (C) 1995 SCI Group


itcl_class VTRDialW {
    inherit BaseDialW

    public frameCommand ""
    public shuttleCommand ""

    constructor {config} {
	BaseDialW::constructor

	$c bind ${basetag}Dial <ButtonRelease-3> "$this b3DialRelease"
	$c bind ${basetag}Dial <1> "$this b1DialDown %x %y"
	$c bind ${basetag}Dial <B1-Motion> "$this b1DialMove %x %y" 
	$c bind ${basetag}Dial <ButtonRelease-1> \
		"$this b1DialRelease %x %y"
    }

    # B3 in the dial switches modes -- spring or crank
    method b3DialRelease {} {
	toggle
	if {$mode == "out"} {
	    setPos 0
	}
	set leftover 0
    }

    # When B1 is pressed, the dial starts tracking where the user moves 
    # the mouse
    method b1DialDown {x y} {
	startTrackMouse $x $y
    }

    # Handle B1 motion differently depending on whether we're in spring or
    # crank mode
    method b1DialMove {x y} {
	if {$mode == "in"} {
	    set stops 0
	} else {
	    set stops 1
	}
	set ll [moveTrackMouse $x $y $stops]
	if {$mode == "in"} {
	    dialRot [lindex $ll 0]
	} else {
	    dialPos [lindex $ll 1]
	}
    }

    # B1 release.  Tell the dial not to track motion any more, and if we're
    # in spring mode, go back to the top
    method b1DialRelease {x y} {
	endTrackMouse $x $y
	if {$mode == "out"} {
	    $c itemconfigure fwdSh -fill #101010
	    $c itemconfigure revSh -fill #101010
	    set shuttleDir 0
	    setPos 0
	    set leftover 0
	}
    }

    # Fractional frames left over while updating LED time
    protected leftover 0

    # Dial is in crank mode -- this gets called with the most recent
    # positional change in radians
    method dialRot {rad} {
	set f [expr $rad*4.77+$leftover]
	set leftover [expr $f-floor($f)]

	if {$frameCommand != ""} {
	    eval $frameCommand [expr floor($f)]
	}
    }

    # Are we shuttling forward (1), backward (-1), or are we stopped (0)
    protected shuttleDir 0

    # Dial is in spring mode -- this gets called with the most recent
    # position. -1 is all the way in reverse, 1 is all the way forward, 0 is
    # stopped.
    method dialPos {pos} {
	if {$pos > 0.01 && $shuttleDir != 1} {
	    $c itemconfigure fwdSh -fill green
	    $c itemconfigure revSh -fill #101010
	    set shuttleDir 1
	}
	if {$pos < -0.01 && $shuttleDir != -1} {
	    $c itemconfigure fwdSh -fill #101010
	    $c itemconfigure revSh -fill green
	    set shuttleDir -1
	}
	if {$pos < 0.01 && $pos > -0.01 && $shuttleDir != 0} {
	    $c itemconfigure fwdSh -fill #101010
	    $c itemconfigure revSh -fill #101010
	    set shuttleDir 0
	}
	
	if {$shuttleCommand != ""} {
	    eval $shuttleCommand $pos
	}
    }
}
