proc makeColorPicker {w var command cancel} {
    global $var

    frame $w.c

    frame $w.c.col -relief ridge -borderwidth 4 -height 1.5c -width 6c \
	    -background #000000
    set col $w.c.col

    frame $w.c.picks
    set picks $w.c.picks

    frame $picks.rgb -relief groove -borderwidth 4
    set rgb $picks.rgb
    fscale $rgb.s1 -label Red -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable r,$w
    fscale $rgb.s2 -label Green -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable g,$w
    fscale $rgb.s3 -label Blue -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable b,$w
    pack $rgb.s1 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s2 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s3 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y

    frame $picks.hsv -relief groove -borderwidth 4 
    set hsv $picks.hsv
    fscale $hsv.s1 -label Hue -from 0.0 -to 360.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable h,$w
    fscale $hsv.s2 -label Saturation -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable s,$w
    fscale $hsv.s3 -label Value -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable v,$w
    pack $hsv.s1 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s2 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s3 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y

    $rgb.s1 configure -command "cpsetrgb $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s2 configure -command "cpsetrgb $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s3 configure -command "cpsetrgb $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s1 configure -command "cpsethsv $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s2 configure -command "cpsethsv $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s3 configure -command "cpsethsv $w $var $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "

    frame $w.c.opts
    button $w.c.opts.ok -text OK -command $command
    button $w.c.opts.cancel -text Cancel -command $cancel
    checkbutton $w.c.opts.rgb -text RGB -variable rgb,$w \
	    -command "cptogrgb $w $picks $rgb"
    checkbutton $w.c.opts.hsv -text HSV -variable hsv,$w \
	    -command "cptoghsv $w $picks $hsv"
    pack $w.c.opts.ok -in $w.c.opts -side left -padx 2 -pady 2 -anchor w
    pack $w.c.opts.cancel -in $w.c.opts -side left -padx 2 -pady 2 -anchor w
    pack $w.c.opts.rgb -in $w.c.opts -side left -padx 2 -pady 2 -anchor e
    pack $w.c.opts.hsv -in $w.c.opts -side left -padx 2 -pady 2 -anchor e

    global rgb,$w hsv,$w
    if [expr ([set rgb,$w] == 1) || ([set hsv,$w] != 1)] {
	set rgb,$w 1
	pack $rgb -in $picks -side left -padx 2 -pady 2 -expand 1 -fill x
    }
    if [expr [set hsv,$w] == 1] {
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

proc cpsetrgb {w var col rs gs bs hs ss vs val} {
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

    set r [expr int([$rs get] * 65535)]
    set g [expr int([$gs get] * 65535)]
    set b [expr int([$bs get] * 65535)]
    
    cpsetcol $var $col [format #%04x%04x%04x $r $g $b]

    update idletasks
}

proc cpsethsv {w var col rs gs bs hs ss vs val} {
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

    set r [expr int([$rs get] * 65535)]
    set g [expr int([$gs get] * 65535)]
    set b [expr int([$bs get] * 65535)]

    cpsetcol $var $col [format #%04x%04x%04x $r $g $b]

    update idletasks
}

proc cpsetcol {var col color} {
    global $var

    $col config -background $color
    set $var $color
}

proc cptogrgb {w picks rgb} {
    global rgb,$w

    if [expr [set rgb,$w] == 1] {
	pack $rgb -in $picks -side left
    } else {
	pack forget $rgb
    }
}

proc cptoghsv {w picks hsv} {
    global hsv,$w

    if [expr [set hsv,$w] == 1] {
	pack $hsv -in $picks -side left
    } else {
	pack forget $hsv
    }
}

