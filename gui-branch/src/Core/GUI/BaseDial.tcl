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

#  BaseDial.tcl
#  Written by:
#   James Purciful and Dave Weinstein
#   Department of Computer Science
#   University of Utah
#   May 1995
#  Copyright (C) 1995 SCI Group


itcl_class BaseDialW {

    public borderWidth 5 {draw}
    public initSize ""     

    constructor {config} {
	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class
	::rename $this $this-win-
	::rename $this-tmp- $this

	frame $this.ui
	canvas $this.ui.canvas 
	set c "$this.ui.canvas"
	pack $this.ui.canvas -in $this.ui -side top -fill both -expand yes
	pack $this.ui -side top -padx 2 -pady 2 -fill both -expand yes

	bind $this.ui.canvas <Configure> "$this setsize %w %h"

	set basetag [nexttag]
	set initialized "yes"

	if [expr ([string compare $initSize ""]!=0)] {
	    setsize $initSize $initSize
	}
    }

    method config {config} {}

    protected oldrad 0
    protected oldx 0
    protected oldy 0

    method setsize {w h config} {
	set mindem [Min $w $h]
	$c config -scrollregion [list 0 0 $mindem $mindem] \
		-width $mindem -height $mindem
	set half [expr int($mindem/2)]
	set rad [expr $half-$borderWidth-1]
	set x [expr int($w/2)]
	set y [expr int($h/2)]
	set xbtn $x
	set ybtn [expr $y-$rad/2]
	set rbtn [expr $rad/6]
	if {$oldrad != $rad || $oldx != $x || $oldy != $y} {
	    draw
	}
	set oldrad $rad
	set oldx $x
	set oldy $y
    }

    method draw {} {
	if [expr ([string compare $initialized "no"]==0)] {
	    return
	}

	$c delete ${basetag}In ${basetag}Out

	mkDial $c $x $y $rad $raised [list ${basetag}Out ${basetag}Dial]
	mkDial $c $xbtn $ybtn $rbtn [expr $raised*-1] \
		[list ${basetag}Out ${basetag}OutBtn ${basetag}Dial]
	mkDial $c $x $y $rad [expr $raised*-1] [list ${basetag}In ${basetag}Dial]
	mkDial $c $xbtn $ybtn $rbtn $raised \
		[list ${basetag}In ${basetag}InBtn ${basetag}Dial]
	$c raise ${basetag}Out ${basetag}In
    }

    # makes a dial widget in canvas c at location (x,y) w/ radius r.
    # raised {1,-1} tells us if it's raised or sunken [i.e. up or down].
    # tgs is the taglist for the canvas item
    method mkDial {c x y r raised tgs} {
	set ty [expr $y-$r]
	set by [expr $y+$r]
	set rx [expr $x+$r]
	set lx [expr $x-$r]
	set ddegree [expr 50/[expr int(sqrt($r))]]
	set grey_start 0
	set grey_stop 225
	set dgrey [expr ($grey_stop-$grey_start)/(180/$ddegree)]
	if {$raised == 1} {
	    set currG $grey_start
	} else {
	    set currG $grey_stop
	    set dgrey [expr $dgrey * -1]
	}
	$c create oval $lx $ty $rx $by -fill #909090 -width 0 -tags $tgs
	set cnt 0
	set deg 0
	while {$deg < 180} {
	    $c create arc $lx $ty $rx $by \
		    -start [expr -45 + $deg] -extent [expr $ddegree+2] \
		    -style arc -fill [rgbColor $currG] \
		    -width $borderWidth -tags $tgs
	    $c create arc $lx $ty $rx $by \
		    -start [expr -45 - $deg] -extent [expr -$ddegree-2] \
		    -style arc -fill [rgbColor $currG] \
		    -width $borderWidth -tags $tgs
	    set currG [expr $currG + $dgrey]
	    set deg [expr $deg + $ddegree]
	}
    }

    method getLine {xp yp} {
	return [list [expr $y-$yp] [expr $xp-$x] \
		[expr ($yp-$y)*$x+($x-$xp)*$y]]
    }

    method toggle {} {
	set raised [expr $raised*-1]
	if {$raised==-1} {
	    set mode "in"
	    $c lower ${basetag}Out ${basetag}In
	} else {
	    set mode "out"
	    $c lower ${basetag}In ${basetag}Out
	}
    }

    # This rotates the dial to the indicated position.
    # The new position is given as 0-1 with 0 as
    # straight up, clockwise rotation.
    method setPos {newpos} {
	set np $newpos
	while {$pos < 0} {
	    set pos [expr $pos+6.2831]
	}
	while {$pos > 6.2831} {
	    set pos [expr $pos-6.2831]
	}
	while {$np < 0} {
	    set np [expr $np+1]
	}
	while {$np > 1} {
	    set np [expr $np-1]
	}
	set chPos [expr (6.2831*$np-$pos)]
	if {$chPos < 0} {
	    set cnt [expr floor([expr $chPos/-.3]+1)]
	    set dpos [expr $chPos/$cnt]
	} else {
	    set cnt [expr floor([expr $chPos/.3]+1)]
	    set dpos [expr $chPos/$cnt]
	}
	while {$cnt > 0} {
	    set pp [expr $pos+$dpos]
	    set sx [expr $x+sin($pp)*$rad/2]
	    set sy [expr $y+cos($pp)*$rad/2]
	    set dx [expr $sx-$xbtn]
	    set dy [expr $sy-$ybtn]
	    $c move ${basetag}OutBtn $dx $dy
	    $c move ${basetag}InBtn $dx $dy
	    update idletasks
	    set xbtn $sx
	    set ybtn $sy
	    set pos $pp
	    set cnt [expr $cnt-1]
	}
    }

    method startTrackMouse {startx starty} {
	set spinx $startx
	set spiny $starty
	set ll [getLine $spinx $spiny]
	set lna [lindex $ll 0]
	set lnb [lindex $ll 1]
	set lnc [lindex $ll 2]
	set x1 [expr $spinx-$x]
	set y1 [expr $spiny-$y]
	set len [expr sqrt([expr $x1*$x1+$y1*$y1])]
	if {$len < .01} {
	    set x1 1
	    set y1 0
	} else {
	    set x1 [expr $x1/$len]
	    set y1 [expr $y1/$len]
	}
	set totaldx 0
	set totaldy 0
	set totalrot 0
    }

    protected side 1
    protected stopped 0
    method moveTrackMouse {newx newy stops} {
	set x2 [expr $newx-$x]
	set y2 [expr $newy-$y]
	set l2 [expr sqrt([expr $x2*$x2+$y2*$y2])]
	if {$l2 < .05} {
	    return
	}
	set x2 [expr $x2/$l2]
	set y2 [expr $y2/$l2]
	set rot [expr acos([expr $x1*$x2+$y1*$y2])] 
	set flip 0
	if {[expr $lna*$newx+$lnb*$newy+$lnc] > 0} {
	    set rot -$rot
	    if {$side == 1} {
		set flip 1
	    }
	    set side -1
	} else {
	    if {$side == -1} {
		set flip 1
	    }
	    set side 1
	}
	if {!$flip} {
	    set stopped 0
	}

	if {$stops && $flip} {
	    if {$rot > 1.5 || $stopped==-1} {
		set stopped -1
		set rot -3.14
		set side -1
	    } else {
		if {$rot < -1.5 || $stopped==1} {
		    set stopped 1
		    set rot 3.14
		    set side 1
		}
	    }
	}
	set pp [expr $rot+$pos]
	set sx [expr $x+sin($pp)*$rad/2]
	set sy [expr $y+cos($pp)*$rad/2]
	set dx [expr $sx-$xbtn]
	set dy [expr $sy-$ybtn]
	set mvdx [expr $dx-$totaldx]
	set mvdy [expr $dy-$totaldy]
	set mvrot [expr $rot-$totalrot]
	while {$mvrot>3.2} {
	    set mvrot [expr $mvrot-6.2832]
	}
	while {$mvrot<-3.2} {
	    set mvrot [expr $mvrot+6.2832]
	}
	set totaldx [expr $totaldx+$mvdx]
	set totaldy [expr $totaldy+$mvdy]
	set totalrot [expr $totalrot+$mvrot]
	$c move ${basetag}OutBtn $mvdx $mvdy
	$c move ${basetag}InBtn $mvdx $mvdy
	return [list [expr $mvrot*-1] [expr (3.141592-$pp)/3.141592]]
    }

    protected pp

    method endTrackMouse {newx newy} {
	set stopped 0
	set sx [expr $x+sin($pp)*$rad/2]
	set sy [expr $y+cos($pp)*$rad/2]
	set dx [expr $sx-$xbtn]
	set dy [expr $sy-$ybtn]
	set mvdx [expr $dx-$totaldx]
	set mvdy [expr $dy-$totaldy]
	$c move ${basetag}OutBtn $mvdx $mvdy
	set pos $pp
	while {$pos < 0} {
	    set pos [expr $pos+6.2831]
	}
	while {$pos > 6.2831} {
	    set pos [expr $pos-6.2831]
	}
	set xbtn $sx
	set ybtn $sy
    }

    common tagid 0

    proc nexttag {} {
	incr tagid 1
	return "tag$tagid"
    }

    method Min {x1 x2} {
	if [expr $x1 < $x2] {
	    return $x1
	} else {
	    return $x2
	}
    }

    protected initialized no

    protected c
    protected basetag
    protected x
    protected y
    protected rad
    protected w
    protected mode out
    protected raised 1
    protected pos 3.141592
    protected spinx
    protected spiny
    protected lna
    protected lnb
    protected lnc
    protected x1
    protected y1
    protected xbtn
    protected ybtn
    protected rbtn
    protected totaldx
    protected totaldy
    protected totalrot
}


proc rgbColor {g} {
    return [format #%02x%02x%02x $g $g $g]
}
