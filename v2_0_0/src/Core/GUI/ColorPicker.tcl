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

#  ColorPicker.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Mar. 1994
#  Copyright (C) 1994 SCI Group

proc makeColorPicker {w var command cancel} {
    global $var

    global $var-r $var-g $var-b $var-a
    global $w-r $w-g $w-b $w-a
    global $w-rgbhsv

    set $w-rgbhsv "rgb"

    set $w-r [set $var-r]
    set $w-g [set $var-g]
    set $w-b [set $var-b]
    if [info exists $var-a] {
	set $w-a [set $var-a]
    }

    frame $w.c

    set ir [expr int([set $w-r] * 65535)]
    set ig [expr int([set $w-g] * 65535)]
    set ib [expr int([set $w-b] * 65535)]
    frame $w.c.col -relief ridge -borderwidth 4 -height 1.5c -width 6c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
    set col $w.c.col

    frame $w.c.picks
    set picks $w.c.picks

    frame $picks.rgb -relief groove -borderwidth 4
    set rgb $picks.rgb
    scale $rgb.s1 -label Red -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-r
    scale $rgb.s2 -label Green -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-g
    scale $rgb.s3 -label Blue -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-b
    pack $rgb.s1 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s2 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s3 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    $rgb.s1 set [set $w-r]
    $rgb.s2 set [set $w-g]
    $rgb.s3 set [set $w-b]

    if [info exists $var-a] {
	scale $rgb.s4 -label Alpha -from 0.0 -to 1.0 -length 6c \
		-showvalue true -orient horizontal -resolution .01 \
		-digits 3 -variable $w-a
	pack $rgb.s4 -in $picks.rgb -side top -padx 2 -pady 2 \
		-anchor nw -fill y
	$rgb.s4 set [set $w-a]
    }

    frame $picks.hsv -relief groove -borderwidth 4 
    set hsv $picks.hsv
    scale $hsv.s1 -label Hue -from 0.0 -to 360.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-h
    scale $hsv.s2 -label Saturation -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-s
    scale $hsv.s3 -label Value -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-v
    pack $hsv.s1 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s2 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s3 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y

    if [info exists $var-a] {
	scale $hsv.s4 -label Alpha -from 0.0 -to 1.0 -length 6c \
		-showvalue true -orient horizontal -resolution .01 \
		-digits 3 -variable $w-a
	pack $hsv.s4 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw \
		-fill y
    }

    $rgb.s1 configure -command "cpsetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s2 configure -command "cpsetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s3 configure -command "cpsetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s1 configure -command "cpsethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s2 configure -command "cpsethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s3 configure -command "cpsethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "

    frame $w.c.opts
    button $w.c.opts.ok -text OK -command "cpcommitcolor $var $rgb.s1 $rgb.s2 $rgb.s3 $rgb.s4 \"$command\""
    button $w.c.opts.cancel -text Cancel -command $cancel
    radiobutton $w.c.opts.rgb -text RGB -variable $w-rgbhsv -value rgb \
	    -command "cptogrgbhsv $w $picks $rgb $hsv"
    radiobutton $w.c.opts.hsv -text HSV -variable $w-rgbhsv -value hsv \
	    -command "cptogrgbhsv $w $picks $rgb $hsv"
    pack $w.c.opts.ok -in $w.c.opts -side left -padx 2 -pady 2 -anchor w
    pack $w.c.opts.cancel -in $w.c.opts -side left -padx 2 -pady 2 -anchor w
    pack $w.c.opts.rgb -in $w.c.opts -side left -padx 2 -pady 2 -anchor e
    pack $w.c.opts.hsv -in $w.c.opts -side left -padx 2 -pady 2 -anchor e


    if { [set $w-rgbhsv] == "rgb" } {
	pack $rgb -in $picks -side left -padx 2 -pady 2 -expand 1 -fill x
    }
    if { [set $w-rgbhsv] == "hsv" } {
	pack $hsv -in $picks -side left -padx 2 -pady 2 -expand 1 -fill x
    }

    pack $w.c.opts $picks $col -in $w.c -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
    pack $w.c
}

proc Max {n1 n2 n3} {
    if [expr $n1 >= $n2] {
	if [expr $n1 >= $n3] {
	    return $n1
	} else {
	    return $n3
	}
    } else {
	if [expr $n2 >= $n3] {
	    return $n2
	} else {
	    return $n3
	}
    }
}

proc Min {n1 n2 n3} {
    if [expr $n1 <= $n2] {
	if [expr $n1 <= $n3] {
	    return $n1
	} else {
	    return $n3
	}
    } else {
	if [expr $n2 <= $n3] {
	    return $n2
	} else {
	    return $n3
	}
    }
}

proc cpsetrgb {col rs gs bs hs ss vs val} {
    # Do inverse transformation to HSV
    set max [Max [$rs get] [$gs get] [$bs get]]
    set min [Min [$rs get] [$gs get] [$bs get]]
    # $ss set [expr ($max == 0.0) ? 0.0 : (($max-$min)/$max)]
    if {$max == 0.0} {
	$ss set 0.0
    } else {
	$ss set [expr ($max-$min)/$max]
    }
    if [expr [$ss get] != 0.0] {
	set rl [expr ($max-[$rs get])/($max-$min)]
	set gl [expr ($max-[$gs get])/($max-$min)]
	set bl [expr ($max-[$bs get])/($max-$min)]
	if [expr $max == [$rs get]] {
	    if [expr $min == [$gs get]] {
		$hs set [expr 60.0*(5.0+$bl)]
	    } else {
		$hs set [expr 60.0*(1.0-$gl)]
	    }
	} elseif [expr $max == [$gs get]] {
	    if [expr $min == [$bs get]] {
		$hs set [expr 60.0*(1.0+$rl)]
	    } else {
		$hs set [expr 60.0*(3.0-$bl)]
	    }
	} else {
	    if [expr $min == [$rs get]] {
		$hs set [expr 60.0*(3.0+$gl)]
	    } else {
		$hs set [expr 60.0*(5.0-$rl)]
	    }
	}
    } else {
	$hs set 0.0
    }
    $vs set $max

    cpsetcol $col [$rs get] [$gs get] [$bs get]

    update idletasks
}

proc cpsethsv {col rs gs bs hs ss vs val} {
    # Convert to RGB...
    while {[$hs get] >= 360.0} {
	$hs set [expr [$hs get] - 360.0]
    }
    while {[$hs get] < 0.0} {
	$hs set [expr [$hs get] + 360.0]
    }
    set h6 [expr [$hs get]/60.0]
    set i [expr int($h6)]
    set f [expr $h6-$i]
    set p1 [expr [$vs get]*(1.0-[$ss get])]
    set p2 [expr [$vs get]*(1.0-([$ss get]*$f))]
    set p3 [expr [$vs get]*(1.0-([$ss get]*(1-$f)))]
    switch $i {
	0 {$rs set [$vs get] ; $gs set $p3 ; $bs set $p1}
	1 {$rs set $p2 ; $gs set [$vs get] ; $bs set $p1}
	2 {$rs set $p1 ; $gs set [$vs get] ; $bs set $p3}
	3 {$rs set $p1 ; $gs set $p2 ; $bs set [$vs get]}
	4 {$rs set $p3 ; $gs set $p1 ; $bs set [$vs get]}
	5 {$rs set [$vs get] ; $gs set $p1 ; $bs set $p2}
	default {$rs set 0 ; $gs set 0 ; $bs set 0}
    }

    cpsetcol $col [$rs get] [$gs get] [$bs get]

    update idletasks
}

proc cpsetcol {col r g b} {
    set ir [expr int($r * 65535)]
    set ig [expr int($g * 65535)]
    set ib [expr int($b * 65535)]

    $col config -background [format #%04x%04x%04x $ir $ig $ib]
}

proc cpcommitcolor {var rs gs bs as command} {
    global $var-r $var-g $var-b $var-a
    set $var-r [$rs get]
    set $var-g [$gs get]
    set $var-b [$bs get]
    if [info exists $var-a] {
	set $var-a [$as get]
    }
    eval $command
}

proc cptogrgbhsv {w picks rgb hsv} {
    global $w-rgbhsv
    if { [set $w-rgbhsv] == "rgb" } {
	pack forget $hsv
	pack $rgb -in $picks -side left
    } else {
	pack forget $rgb
	pack $hsv -in $picks -side left
    }
}
