#
#  MaterialEditor.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Mar. 1995
#
#  Copyright (C) 1995 SCI Group
#

#  These are the components of a Material property:
#    Color  ambient;
#    Color  diffuse;
#    Color  specular;
#    double shininess;
#    Color  emission;
#    double reflectivity;
#    double transparency;
#    double refraction_index;

proc setColor {to from} {
    global $to-r $from-r
    global $to-g $from-g
    global $to-b $from-b
    set $to-r [set $from-r]
    set $to-g [set $from-g]
    set $to-b [set $from-b]
}

proc setMaterial {to from} {
    global $to-shininess $from-shininess
    global $to-reflectivity $from-reflectivity
    global $to-transparency $from-transparency
    global $to-refraction_index $from-refraction_index
    setColor $to-ambient $from-ambient
    setColor $to-diffuse $from-diffuse
    setColor $to-specular $from-specular
    set $to-shininess [set $from-shininess]
    setColor $to-emission $from-emission
    set $to-reflectivity [set $from-reflectivity]
    set $to-transparency [set $from-transparency]
    set $to-refraction_index [set $from-refraction_index]
}

proc makeMaterialEditor {w var command cancel} {
    global $var

    global $var-ambient-r $var-ambient-g $var-ambient-b
    global $var-diffuse-r $var-diffuse-g $var-diffuse-b
    global $var-specular-r $var-specular-g $var-specular-b
    global $var-shininess
    global $var-emission-r $var-emission-g $var-emission-b
    global $var-reflectivity
    global $var-transparency
    global $var-refraction_index
    global $w-ambient-r $w-ambient-g $w-ambient-b
    global $w-diffuse-r $w-diffuse-g $w-diffuse-b
    global $w-specular-r $w-specular-g $w-specular-b
    global $w-shininess
    global $w-emission-r $w-emission-g $w-emission-b
    global $w-reflectivity
    global $w-transparency
    global $w-refraction_index

    setMaterial $w $var

    frame $w.lmr -relief groove -borderwidth 4
    frame $w.lmr.left
    set left $w.lmr.left
    frame $w.lmr.middle
    set middle $w.lmr.middle
    frame $w.lmr.right
    set right $w.lmr.right

    label $left.ambient -text Ambient
    set ir [expr int([set $w-ambient-r] * 65535)]
    set ig [expr int([set $w-ambient-g] * 65535)]
    set ib [expr int([set $w-ambient-b] * 65535)]
    button $left.amb -relief sunken -borderwidth 4 -height 2 -width 34 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global ambient
    set ambient $left.amb

    label $middle.diffuse -text Diffuse
    set ir [expr int([set $w-diffuse-r] * 65535)]
    set ig [expr int([set $w-diffuse-g] * 65535)]
    set ib [expr int([set $w-diffuse-b] * 65535)]
    button $middle.dif -relief sunken -borderwidth 4 -height 2 -width 34 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global diffuse
    set diffuse $middle.dif

    label $right.specular -text Specular
    set ir [expr int([set $w-specular-r] * 65535)]
    set ig [expr int([set $w-specular-g] * 65535)]
    set ib [expr int([set $w-specular-b] * 65535)]
    button $right.spe -relief sunken -borderwidth 4 -height 2 -width 34 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global specular
    set specular $right.spe

    label $right.shiny -text Shininess
    scale $right.shi -from 0.0 -to 128.0 -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-shininess
    set shiny $right.shi

    label $left.emission -text Emission
    set ir [expr int([set $w-emission-r] * 65535)]
    set ig [expr int([set $w-emission-g] * 65535)]
    set ib [expr int([set $w-emission-b] * 65535)]
    button $left.emi -relief sunken -borderwidth 4 -height 2 -width 34 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global emission
    set emission $left.emi

    label $middle.reflect -text Reflectivity
    scale $middle.ref -from 0.0 -to 1.0 -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-reflectivity
    set reflect $middle.ref

    label $left.transp -text Transparency
    scale $left.tra -from 0.0 -to 1.0 -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-transparency
    set transp $left.tra

    label $middle.refract -text "Refraction Index"
    scale $middle.rin -from 0.5 -to 2.0 -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-refraction_index
    set refract $middle.rin

    pack $left.ambient -in $left -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $ambient -in $left -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $middle.diffuse -in $middle -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $diffuse -in $middle -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $right.specular -in $right -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $specular -in $right -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $right.shiny -in $right -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $shiny -in $right -side top -pady 2 -anchor nw -fill both -expand 1
    pack $left.emission -in $left -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $emission -in $left -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $middle.reflect -in $middle -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $reflect -in $middle -side top -pady 2 -anchor nw -fill both -expand 1
    pack $left.transp -in $left -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $transp -in $left -side top -pady 2 -anchor nw -fill both -expand 1
    pack $middle.refract -in $middle -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $refract -in $middle -side top -pady 2 -anchor nw -fill both -expand 1

    pack $left -in $w.lmr -side left -pady 2 -anchor nw -fill y
    pack $middle -in $w.lmr -side left -pady 2 -anchor nw -fill y
    pack $right -in $w.lmr -side left -pady 2 -anchor nw -fill y
    pack $w.lmr -in $w -side top -padx 4 -pady 4 -expand 1 -fill both

    frame $w.material
    frame $w.material.color -relief groove -borderwidth 4
    set material $w.material.color

    set ir [expr int([set $w-ambient-r] * 65535)]
    set ig [expr int([set $w-ambient-g] * 65535)]
    set ib [expr int([set $w-ambient-b] * 65535)]
    frame $material.col -relief sunken -borderwidth 4 -height 1.5c -width 6c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
    set col $material.col

    frame $material.picks
    set picks $material.picks

    frame $picks.rgb -relief groove -borderwidth 4
    set rgb $picks.rgb
    scale $rgb.s1 -label Red -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-r
    scale $rgb.s2 -label Green -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-g
    scale $rgb.s3 -label Blue -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-b
    pack $rgb.s1 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s2 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $rgb.s3 -in $picks.rgb -side top -padx 2 -pady 2 -anchor nw -fill y
    $rgb.s1 set [set $w-ambient-r]
    $rgb.s2 set [set $w-ambient-g]
    $rgb.s3 set [set $w-ambient-b]

    frame $picks.hsv -relief groove -borderwidth 4 
    set hsv $picks.hsv
    scale $hsv.s1 -label Hue -from 0.0 -to 360.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-h
    scale $hsv.s2 -label Saturation -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-s
    scale $hsv.s3 -label Value -from 0.0 -to 1.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-v
    pack $hsv.s1 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s2 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y
    pack $hsv.s3 -in $picks.hsv -side top -padx 2 -pady 2 -anchor nw -fill y

    $rgb.s1 configure -command "mesetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s2 configure -command "mesetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $rgb.s3 configure -command "mesetrgb $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s1 configure -command "mesethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s2 configure -command "mesethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "
    $hsv.s3 configure -command "mesethsv $col $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3 "

    $ambient configure -command "meset $w $col 0 $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3"
    $diffuse configure -command "meset $w $col 1 $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3"
    $specular configure -command "meset $w $col 2 $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3"
    $emission configure -command "meset $w $col 3 $rgb.s1 $rgb.s2 $rgb.s3 \
	    $hsv.s1 $hsv.s2 $hsv.s3"

    frame $material.opts
    button $material.opts.replace -text Replace -command "mecommitcolor $w $rgb.s1 $rgb.s2 $rgb.s3"
    global $w-i
    set $w-i 0
    radiobutton $material.opts.amb -text Ambient -value 0 -variable $w-i \
	    -command "meset $w $col 0 $rgb.s1 $rgb.s2 $rgb.s3 $hsv.s1 $hsv.s2 $hsv.s3"
    radiobutton $material.opts.dif -text Diffuse -value 1 -variable $w-i \
	    -command "meset $w $col 1 $rgb.s1 $rgb.s2 $rgb.s3 $hsv.s1 $hsv.s2 $hsv.s3"
    radiobutton $material.opts.spe -text Specular -value 2 -variable $w-i \
	    -command "meset $w $col 2 $rgb.s1 $rgb.s2 $rgb.s3 $hsv.s1 $hsv.s2 $hsv.s3"
    radiobutton $material.opts.emi -text Emission -value 3 -variable $w-i \
	    -command "meset $w $col 3 $rgb.s1 $rgb.s2 $rgb.s3 $hsv.s1 $hsv.s2 $hsv.s3"
    pack $material.opts.replace -in $material.opts -side left -padx 2 -pady 2 -anchor w
    pack $material.opts.amb -in $material.opts -side left -padx 2 -pady 2 -anchor w
    pack $material.opts.dif -in $material.opts -side left -padx 2 -pady 2 -anchor w
    pack $material.opts.spe -in $material.opts -side left -padx 2 -pady 2 -anchor w
    pack $material.opts.emi -in $material.opts -side left -padx 2 -pady 2 -anchor w

    pack $rgb -in $picks -side left -padx 2 -pady 2 -expand 1 -fill x
    pack $hsv -in $picks -side left -padx 2 -pady 2 -expand 1 -fill x

    pack $material.opts $picks $col -in $material -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
    pack $material

    frame $w.material.sample -relief groove -borderwidth 4
    frame $w.material.sample.opts
    set opts $w.material.sample.opts

    button $opts.ok -text OK -command "mecommit $w $var \"$command\""
    button $opts.cancel -text Cancel -command $cancel
    button $opts.update -text Update -command "puts \"Preview not implemented!\""
    pack $opts.ok -in $opts -side left -padx 2 -pady 2 -anchor w
    pack $opts.cancel -in $opts -side left -padx 2 -pady 2 -anchor w
    pack $opts.update -in $opts -side top -padx 2 -pady 2 -anchor nw
    pack $opts -in $w.material.sample -side top -padx 2 -pady 2 -expand 1 -fill both

    canvas $w.material.sample.sam -width 7c -height 7c -background #000000
    pack $w.material.sample.sam -in $w.material.sample -side left -padx 2 -pady 2 -anchor nw
    $w.material.sample.sam create text 2.6c 3c -text Preview -anchor sw -fill white
    $w.material.sample.sam create text 3c 3.5c -text Not -anchor sw -fill red
    $w.material.sample.sam create text 2c 4c -text Implemented -anchor sw -fill white

    pack $w.material.sample -in $w.material -side left -padx 2 -pady 2 -anchor nw
    pack $w.material.color -in $w.material -side left -padx 4 -pady 2 -anchor nw
    pack $w.material -in $w -side top -padx 2 -pady 2 -anchor nw -fill y
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

proc mesetrgb {col rs gs bs hs ss vs val} {
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

    mesetcol $col [$rs get] [$gs get] [$bs get]

    update idletasks
}

proc mesethsv {col rs gs bs hs ss vs val} {
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

    mesetcol $col [$rs get] [$gs get] [$bs get]

    update idletasks
}

proc mesetcol {col r g b} {
    set ir [expr int($r * 65535)]
    set ig [expr int($g * 65535)]
    set ib [expr int($b * 65535)]

    $col config -background [format #%04x%04x%04x $ir $ig $ib]
}

proc meset {w col i rs gs bs hs ss vs} {
    global $w-i
    set $w-i $i
    
    switch [set $w-i] {
	0 {
	    set color $w-ambient
	}
	1 {
	    set color $w-diffuse
	}
	2 {
	    set color $w-specular
	}
	3 {
	    set color $w-emission
	}
	default {
	    puts "Unknown color type!" ;
	    set color $w-ambient
	}
    }    

    global $color-r $color-g $color-b
    $rs set [set $color-r]
    $gs set [set $color-g]
    $bs set [set $color-b]
    mesetrgb $col $rs $gs $bs $hs $ss $vs 0.0
}

proc mecommitcolor {w rs gs bs} {
    set r [$rs get]
    set g [$gs get]
    set b [$bs get]
    set ir [expr int($r * 65535)]
    set ig [expr int($g * 65535)]
    set ib [expr int($b * 65535)]

    global $w-i
    switch [set $w-i] {
	0 {
	    global ambient ;
	    set col $w-ambient ;
	    set mattype $ambient
	}
	1 {
	    global diffuse ;
	    set col $w-diffuse ;
	    set mattype $diffuse
	}
	2 {
	    global specular ;
	    set col $w-specular ;
	    set mattype $specular
	}
	3 {
	    global emission ;
	    set col $w-emission ;
	    set mattype $emission
	}
	default {
	    puts "Unknown color type!" ;
	    set ir [expr int(0)] ;
	    set ig [expr int(0)] ;
	    set ib [expr int(0)] ;
	    global ambient ;
	    set col $w-ambient ;
	    set mattype $ambient
	}
    }

    global $col-r $col-g $col-b
    set $col-r $r
    set $col-g $g
    set $col-b $b
    $mattype config -background [format #%04x%04x%04x $ir $ig $ib]
    $mattype config -activebackground [format #%04x%04x%04x $ir $ig $ib]
}

proc mecommit {w var command} {
    global $var $w
    setMaterial $var $w
    eval $command
}
