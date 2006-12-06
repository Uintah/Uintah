# range.tcl --
#
# This file defines the default bindings for Tk range widgets.
#
# @(#) rangxbe.tcl 1.3 94/12/17 16:05:23
#
# Copyright (c) 1994 The Regents of the University of California.
# Copyright (c) 1994 Sun Microsystems, Inc.
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
#
# Originally based on the tkRange widget, and then totally modified by...
# David Weinstein 
# February 1995
# Copyright SCI
#

# tkRangeBind --
# This procedure below invoked the first time the mouse enters a
# range widget or a range widget receives the input focus.  It creates
# all of the class bindings for ranges.
#
# Arguments:
# event -       Indicates which event caused the procedure to be invoked
#               (Enter or FocusIn).  It is used so that we can carry out
#               the functions of that event in addition to setting up
#               bindings.

 bind Range <FocusIn> {}

 bind Range <Enter> {
     if $tk_strictMotif {
 	set tkPriv(activeBg) [%W cget -activebackground]
 	%W config -activebackground [%W cget -background]
     }
     tkRangeActivate %W %x %y
 }

 # bind Range <Motion> {
 #    tkRangeActivate %W %x %y
 # }

 bind Range <Leave> {
     if $tk_strictMotif {
 	%W config -activebackground $tkPriv(activeBg)
     }
     if {[%W cget -state] == "active"} {
 	%W configure -state normal
     }
 }
 bind Range <1> {
     tkRangeButtonDown %W %x %y
 }
 bind Range <B1-Motion> {
     tkRangeDrag %W %x %y
 }
 bind Range <B1-Leave> { }
 bind Range <B1-Enter> { }
 bind Range <ButtonRelease-1> {
     tkCancelRepeat
     tkRangeEndDrag %W
     tkRangeActivate %W %x %y
 }
 bind Range <2> {
     tkRangeButton2Down %W %x %y
 }
 bind Range <B2-Motion> {
     tkRangeDrag %W %x %y
 }
 bind Range <B2-Leave> { }
 bind Range <B2-Enter> { }
 bind Range <ButtonRelease-2> {
     tkCancelRepeat
     tkRangeEndDrag %W
     tkRangeActivate %W %x %y
 }
 bind Range <Control-1> {
     tkRangeControlPress %W %x %y
 }
 bind Range <Up> {
     tkRangeIncrement %W up little noRepeat
 }
 bind Range <Down> {
     tkRangeIncrement %W down little noRepeat
 }
 bind Range <Left> {
     tkRangeIncrement %W up little noRepeat
 }
 bind Range <Right> {
     tkRangeIncrement %W down little noRepeat
 }
 bind Range <Control-Up> {
     tkRangeIncrement %W up big noRepeat
 }
 bind Range <Control-Down> {
     tkRangeIncrement %W down big noRepeat
 }
 bind Range <Control-Left> {
     tkRangeIncrement %W up big noRepeat
 }
 bind Range <Control-Right> {
     tkRangeIncrement %W down big noRepeat
 }
 bind Range <Home> {
     %W set [%W cget -from]
 }
 bind Range <End> {
     %W set [%W cget -to]
 }

 # tkRangeActivate --
 # This procedure is invoked to check a given x-y position in the
 # range and activate the slider if the x-y position falls within
 # the slider.
 #
 # Arguments:
 # w -           The range widget.
 # x, y -        Mouse coordinates.

 proc tkRangeActivate {w x y} {
     global tkPriv
     if {[$w cget -state] == "disabled"} {
         return;
     }
     if {[$w identify $x $y] == "slider"} {
         $w configure -state active
     } else {
         $w configure -state normal
     }
 }

 # tkRangeButtonDown --
 # This procedure is invoked when a button is pressed in a range.  It
 # takes different actions depending on where the button was pressed.
 #
 # Arguments:
 # w -           The range widget.
 # x, y -        Mouse coordinates of button press.

 proc tkRangeButtonDown {w x y} {
     global tkPriv
     set tkPriv(dragging) 0
     set el [$w identify $x $y]
     if {$el == "trough1"} {
         set tkPriv(dragging) "min"
         set coords [$w coordsMin]
         set tkPriv(deltaX) [expr $x - [lindex $coords 0]]
         set tkPriv(deltaY) [expr $y - [lindex $coords 1]]

  	 tkRangeIncrement $w up little initial
     } elseif {$el == "trough2"} {
         set tkPriv(dragging) "max"
         set coords [$w coordsMax]
         set tkPriv(deltaX) [expr $x - [lindex $coords 0]]
         set tkPriv(deltaY) [expr $y - [lindex $coords 1]]

  	 tkRangeIncrement $w down little initial
     } elseif {$el == "range"} {
         set tkPriv(dragging) "range"
         set tkPriv(initValue) [$w get $x $y]
     } elseif {$el == "min_slider"} {
         set coords [$w coordsMin]
         set tkPriv(deltaX) [expr $x - [lindex $coords 0]]
         set tkPriv(deltaY) [expr $y - [lindex $coords 1]]
         set tkPriv(dragging) "min"
     } elseif {$el == "max_slider"} {
         set coords [$w coordsMax]
         set tkPriv(deltaX) [expr $x - [lindex $coords 0]]
         set tkPriv(deltaY) [expr $y - [lindex $coords 1]]
         set tkPriv(dragging) "max"
     }
 }

 # tkRangeDrag --
 # This procedure is called when the mouse is dragged with
 # mouse button 1 or 2 down.  If the drag started inside the slider
 # (i.e. the range is active) then the range's value is adjusted
 # to reflect the mouse's position.
 #
 # Arguments:
 # w -           The range widget.
 # x, y -        Mouse coordinates.

 proc tkRangeDrag {w x y} {
     global tkPriv
     if {$tkPriv(dragging) == 0} {
         return
     }
     if {$tkPriv(dragging) == "min"} {
         set val [$w get [expr $x - $tkPriv(deltaX)] \
                  [expr $y - $tkPriv(deltaY)]]
         if {$val > [$w to]} {
             set val [$w to]
         }
         if {$val > [$w getMax]} {
             $w setMinMax $val $val
         } else {
             $w setMin $val
         }
     } elseif {$tkPriv(dragging) == "max"} {
         set val [$w get [expr $x - $tkPriv(deltaX)] \
                  [expr $y - $tkPriv(deltaY)]]
         if {$val < [$w from]} {
             set val [$w from]
         }
         if {$val < [$w getMin]} {
             $w setMinMax $val $val
         } else {
             $w setMax $val
         }
     } elseif {$tkPriv(dragging) == "range"} {
         set val [$w get $x $y]
         set diff [expr $val - $tkPriv(initValue)]
         if {$diff == 0} return
         set to [$w to]
         set from [$w from]
         set newMin [expr [$w getMin] + $diff]
         set newMax [expr [$w getMax] + $diff]
         if {$newMin >= $from && $newMax <= $to} {
             $w setMinMax $newMin $newMax
             set tkPriv(initValue) $val
             return
         }
         if {$newMin < $from} {
             set diffMax [expr $from - [$w getMin]] 
             set newMax [expr [$w getMax] + $diffMax]
             $w setMinMax $from $newMax
             set tkPriv(initValue) $val
         } else {
             set diffMin [expr $to - [$w getMax]]
             set newMin [expr [$w getMin] + $diffMin] 
             $w setMinMax $newMin $to
             set tkPriv(initValue) $val
         }
     }
 }

 # tkRangeEndDrag --
 # This procedure is called to end an interactive drag of the
 # slider.
 #
 # Arguments:
 # w -           The range widget.
 # x, y -        The mouse position at the end of the drag.

 proc tkRangeEndDrag {w} {
     global tkPriv
     if {$tkPriv(dragging) == 0} {
         return
     }
     set tkPriv(dragging) 0
 }

# tkRangeIncrement --
# This procedure is invoked to increment the value of a range and
# to set up auto-repeating of the action if that is desired.  The
# way the value is incremented depends on the "dir" and "big"
# arguments.
#
# Arguments:
# w -		The range widget.
# dir -		"up" means move value towards -from, "down" means
#		move towards -to.
# big -		Size of increments: "big" or "little".
# repeat -	Whether and how to auto-repeat the action:  "noRepeat"
#		means don't auto-repeat, "initial" means this is the
#		first action in an auto-repeat sequence, and "again"
#		means this is the second repetition or later.

proc tkRangeIncrement {w dir big repeat} {
    global tkPriv
    if {![winfo exists $w]} return
    if {[string equal $big "big"]} {
	set inc [$w cget -bigincrement]
	if {$inc == 0} {
	    set inc [expr {abs([$w cget -to] - [$w cget -from])/10.0}]
	}
	if {$inc < [$w cget -resolution]} {
	    set inc [$w cget -resolution]
	}
    } else {
	set inc [$w cget -resolution]
    }
    if {([$w cget -from] > [$w cget -to]) ^ [string equal $dir "up"]} {
	set inc [expr {-$inc}]
    }
    if {[string equal $dir "up"]} {
	$w setMin [expr {[$w getMin] + $inc}]
    } else {
	$w setMax [expr {[$w getMax] + $inc}]
    }

    if {[string equal $repeat "again"]} {
	set tkPriv(afterId) [after [$w cget -repeatinterval] \
		[list tkRangeIncrement $w $dir $big again]]
    } elseif {[string equal $repeat "initial"]} {
	set delay [$w cget -repeatdelay]
	if {$delay > 0} {
	    set tkPriv(afterId) [after $delay \
		    [list tkRangeIncrement $w $dir $big again]]
	}
    }
}