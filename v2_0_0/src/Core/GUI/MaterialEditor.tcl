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

#  MaterialEditor.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Mar. 1995
#  Copyright (C) 1995 SCI Group

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

    global $w-ambient-r $w-ambient-g $w-ambient-b
    global $w-diffuse-r $w-diffuse-g $w-diffuse-b
    global $w-specular-r $w-specular-g $w-specular-b
    global $w-shininess
    global $w-emission-r $w-emission-g $w-emission-b
    global $w-reflectivity
    global $w-transparency
    global $w-refraction_index

    setMaterial $w $var

    frame $w.lmr
    frame $w.lmr.left
    set left $w.lmr.left
    frame $w.lmr.mr -relief groove -borderwidth 4
    frame $w.lmr.mr.middle
    set middle $w.lmr.mr.middle
    frame $w.lmr.mr.right
    set right $w.lmr.mr.right

    label $middle.ambient -text Ambient
    set ir [expr int([set $w-ambient-r] * 65535)]
    set ig [expr int([set $w-ambient-g] * 65535)]
    set ib [expr int([set $w-ambient-b] * 65535)]
    button $right.amb -relief sunken -borderwidth 4 -height 1 -width 17 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global ambient
    set ambient $right.amb

    label $middle.diffuse -text Diffuse
    set ir [expr int([set $w-diffuse-r] * 65535)]
    set ig [expr int([set $w-diffuse-g] * 65535)]
    set ib [expr int([set $w-diffuse-b] * 65535)]
    button $right.dif -relief sunken -borderwidth 4 -height 1 -width 17 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global diffuse
    set diffuse $right.dif

    label $middle.specular -text Specular
    set ir [expr int([set $w-specular-r] * 65535)]
    set ig [expr int([set $w-specular-g] * 65535)]
    set ib [expr int([set $w-specular-b] * 65535)]
    button $right.spe -relief sunken -borderwidth 4 -height 1 -width 17 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global specular
    set specular $right.spe

    label $middle.shiny -text Shininess
    scale $right.shi -from 0.0 -to 128.0 -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-shininess
    set shiny $right.shi

    label $middle.emission -text Emission
    set ir [expr int([set $w-emission-r] * 65535)]
    set ig [expr int([set $w-emission-g] * 65535)]
    set ib [expr int([set $w-emission-b] * 65535)]
    button $right.emi -relief sunken -borderwidth 4 -height 1 -width 17 \
	    -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    global emission
    set emission $right.emi

    label $middle.reflect -text Reflectivity
    scale $right.ref -from 0.0 -to 1.0 -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-reflectivity
    set reflect $right.ref

    label $middle.transp -text Transparency
    scale $right.tra -from 0.0 -to 1.0 -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-transparency
    set transp $right.tra

    label $middle.refract -text "Refraction Index"
    scale $right.rin -from 0.5 -to 2.0 -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $w-refraction_index
    set refract $right.rin

    pack $middle.ambient -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $ambient -in $right -side top -padx 2 -anchor nw -fill y
    pack $middle.diffuse -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $diffuse -in $right -side top -padx 2 -anchor nw -fill y
    pack $middle.specular -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $specular -in $right -side top -padx 2 -anchor nw -fill y
    pack $middle.shiny -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $shiny -in $right -side top -anchor nw -fill both -expand 1
    pack $middle.emission -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $emission -in $right -side top -padx 2 -anchor nw -fill y
    pack $middle.reflect -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $reflect -in $right -side top -anchor nw -fill both -expand 1
    pack $middle.transp -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $transp -in $right -side top -anchor nw -fill both -expand 1
    pack $middle.refract -in $middle -side top -padx 2 -anchor nw -expand y -fill y
    pack $refract -in $right -side top -anchor nw -fill both -expand 1

    frame $w.material
    frame $w.material.color -relief groove -borderwidth 4
    set material $w.material.color

    frame $material.picks
    set picks $material.picks

    frame $picks.rgb -relief groove -borderwidth 4
    frame $picks.rgb.labels
    set labels $picks.rgb.labels
    label $labels.r -text R
    label $labels.g -text G
    label $labels.b -text B
    pack $labels.r -in $labels -side top -padx 2 -pady 2 -anchor nw -expand y -fill y
    pack $labels.g -in $labels -side top -padx 2 -pady 2 -anchor nw -expand y -fill y
    pack $labels.b -in $labels -side top -padx 2 -pady 2 -anchor nw -expand y -fill y
    frame $picks.rgb.sliders
    set rgb $picks.rgb.sliders
    scale $rgb.s1 -from 0.0 -to 1.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-r
    scale $rgb.s2 -from 0.0 -to 1.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-g
    scale $rgb.s3 -from 0.0 -to 1.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-b
    pack $rgb.s1 -in $rgb -side top -padx 2 -anchor nw -fill y
    pack $rgb.s2 -in $rgb -side top -padx 2 -anchor nw -fill y
    pack $rgb.s3 -in $rgb -side top -padx 2 -anchor nw -fill y
    pack $labels -in $picks.rgb -side left -padx 2 -anchor nw -fill y
    pack $rgb -in $picks.rgb -side left -padx 2 -anchor nw -fill y
    pack $picks.rgb -in $picks -side left -padx 2 -anchor nw -expand y -fill y
    $rgb.s1 set [set $w-ambient-r]
    $rgb.s2 set [set $w-ambient-g]
    $rgb.s3 set [set $w-ambient-b]

    frame $picks.hsv -relief groove -borderwidth 4 
    frame $picks.hsv.labels
    set labels $picks.hsv.labels
    label $labels.h -text H
    label $labels.s -text S
    label $labels.v -text V
    pack $labels.h -in $labels -side top -padx 2 -anchor nw -expand y -fill y
    pack $labels.s -in $labels -side top -padx 2 -anchor nw -expand y -fill y
    pack $labels.v -in $labels -side top -padx 2 -anchor nw -expand y -fill y
    frame $picks.hsv.sliders
    set hsv $picks.hsv.sliders
    scale $hsv.s1 -from 0.0 -to 360.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-h
    scale $hsv.s2 -from 0.0 -to 1.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-s
    scale $hsv.s3 -from 0.0 -to 1.0 -length 5c -showvalue true -width 3m \
	    -orient horizontal -resolution .01 \
	    -digits 3 -variable $material-v
    pack $hsv.s1 -in $hsv -side top -padx 2 -anchor nw -fill y
    pack $hsv.s2 -in $hsv -side top -padx 2 -anchor nw -fill y
    pack $hsv.s3 -in $hsv -side top -padx 2 -anchor nw -fill y
    pack $labels -in $picks.hsv -side left -padx 2 -anchor nw -fill y
    pack $hsv -in $picks.hsv -side left -padx 2 -anchor nw -fill y
    pack $picks.hsv -in $picks -side left -padx 2 -anchor nw -expand y -fill y

    frame $material.opts
    set ir [expr int([set $w-ambient-r] * 65535)]
    set ig [expr int([set $w-ambient-g] * 65535)]
    set ib [expr int([set $w-ambient-b] * 65535)]
    frame $material.opts.col -relief sunken -borderwidth 4 -height 8m -width 2.5c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
    set col $material.opts.col

    global $w.rs $w.gs $w.bs
    set $w.rs $rgb.s1
    set $w.gs $rgb.s2
    set $w.bs $rgb.s3
    global $w.hs $w.ss $w.vs
    set $w.hs $hsv.s1
    set $w.ss $hsv.s2
    set $w.vs $hsv.s3
    global $w.col
    set $w.col $col
    global $w.var
    set $w.var $var

    global $w-i
    set $w-i 0
    radiobutton $material.opts.amb -text Ambient -value 0 -variable $w-i \
	    -command "meset $w 0"
    radiobutton $material.opts.dif -text Diffuse -value 1 -variable $w-i \
	    -command "meset $w 1"
    radiobutton $material.opts.spe -text Specular -value 2 -variable $w-i \
	    -command "meset $w 2"
    radiobutton $material.opts.emi -text Emission -value 3 -variable $w-i \
	    -command "meset $w 3"
    button $material.opts.replace -text Replace -command "mecommitcolor $w"
    pack $material.opts.amb $material.opts.dif $material.opts.spe $material.opts.emi $col \
	    -in $material.opts -side left -pady 2 -fill both -anchor w
    pack $material.opts.replace -in $material.opts -side left -padx 2 -pady 2 -anchor w
    pack $picks $material.opts -in $material -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
    pack $material

    pack $w.material.color -in $w.material -side left -padx 2 -pady 2 -anchor nw

    frame $left.sample -relief groove -borderwidth 4
    frame $left.sample.opts
    set opts $left.sample.opts

    button $opts.ok -text OK -command "mecommit $w \"$command\""
    button $opts.cancel -text Cancel -command $cancel
    button $opts.preview -text Preview -command "puts \"Preview not implemented!\""
    button $opts.resync -text Resync -command "meresync $w"
    pack $opts.ok $opts.cancel $opts.preview $opts.resync -in $opts -side left -anchor nw
    pack $opts -in $left.sample -side top -fill both -anchor nw

    canvas $left.sample.sam -width 5.9c -height 5.9c -background #FFFFFF
    pack $left.sample.sam -in $left.sample -side top -padx 2 -pady 2 -expand 1 -fill both -anchor nw
    $left.sample.sam create text 2c 3c -text Preview -anchor sw -fill black
    $left.sample.sam create text 2.5c 3.5c -text Not -anchor sw -fill red
    $left.sample.sam create text 1.5c 4c -text Implemented -anchor sw -fill black
    pack $left.sample -in $left -side left -padx 2 -pady 2 -expand 1 -fill both -anchor nw

    pack $middle $right -in $w.lmr.mr -side left -pady 2 -anchor nw -fill both
    pack $left $w.lmr.mr -in $w.lmr -side left -pady 2 -anchor nw -fill both 
    pack $w.lmr $w.material -in $w -side top -padx 2 -pady 2 -anchor nw -expand 1 -fill both

    $rgb.s1 configure -command "mesetrgb $w "
    $rgb.s2 configure -command "mesetrgb $w "
    $rgb.s3 configure -command "mesetrgb $w "
    $hsv.s1 configure -command "mesethsv $w "
    $hsv.s2 configure -command "mesethsv $w "
    $hsv.s3 configure -command "mesethsv $w "

    $ambient configure -command "meset $w 0"
    $diffuse configure -command "meset $w 1"
    $specular configure -command "meset $w 2"
    $emission configure -command "meset $w 3"

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

proc mesetrgb {w val} {
    global $w.rs $w.gs $w.bs
    global $w.hs $w.ss $w.vs

    # Do inverse transformation to HSV
    set max [Max [[set $w.rs] get] [[set $w.gs] get] [[set $w.bs] get]]
    set min [Min [[set $w.rs] get] [[set $w.gs] get] [[set $w.bs] get]]
    # [set $w.ss] set [expr ($max == 0.0) ? 0.0 : (($max-$min)/$max)]
    if {$max == 0.0} {
	[set $w.ss] set 0.0
    } else {
	[set $w.ss] set [expr ($max-$min)/$max]
    }
    if [expr [[set $w.ss] get] != 0.0] {
	set rl [expr ($max-[[set $w.rs] get])/($max-$min)]
	set gl [expr ($max-[[set $w.gs] get])/($max-$min)]
	set bl [expr ($max-[[set $w.bs] get])/($max-$min)]
	if [expr $max == [[set $w.rs] get]] {
	    if [expr $min == [[set $w.gs] get]] {
		[set $w.hs] set [expr 60.0*(5.0+$bl)]
	    } else {
		[set $w.hs] set [expr 60.0*(1.0-$gl)]
	    }
	} elseif [expr $max == [[set $w.gs] get]] {
	    if [expr $min == [[set $w.bs] get]] {
		[set $w.hs] set [expr 60.0*(1.0+$rl)]
	    } else {
		[set $w.hs] set [expr 60.0*(3.0-$bl)]
	    }
	} else {
	    if [expr $min == [[set $w.rs] get]] {
		[set $w.hs] set [expr 60.0*(3.0+$gl)]
	    } else {
		[set $w.hs] set [expr 60.0*(5.0-$rl)]
	    }
	}
    } else {
	[set $w.hs] set 0.0
    }
    [set $w.vs] set $max

    global $w.col
    mesetcol [set $w.col] [[set $w.rs] get] [[set $w.gs] get] [[set $w.bs] get]

    update idletasks
}

proc mesethsv {w val} {
    global $w.rs $w.gs $w.bs
    global $w.hs $w.ss $w.vs

    # Convert to RGB...
    while {[[set $w.hs] get] >= 360.0} {
	[set $w.hs] set [expr [[set $w.hs] get] - 360.0]
    }
    while {[[set $w.hs] get] < 0.0} {
	[set $w.hs] set [expr [[set $w.hs] get] + 360.0]
    }
    set h6 [expr [[set $w.hs] get]/60.0]
    set i [expr int($h6)]
    set f [expr $h6-$i]
    set p1 [expr [[set $w.vs] get]*(1.0-[[set $w.ss] get])]
    set p2 [expr [[set $w.vs] get]*(1.0-([[set $w.ss] get]*$f))]
    set p3 [expr [[set $w.vs] get]*(1.0-([[set $w.ss] get]*(1-$f)))]
    switch $i {
	0 {[set $w.rs] set [[set $w.vs] get] ; [set $w.gs] set $p3 ; [set $w.bs] set $p1}
	1 {[set $w.rs] set $p2 ; [set $w.gs] set [[set $w.vs] get] ; [set $w.bs] set $p1}
	2 {[set $w.rs] set $p1 ; [set $w.gs] set [[set $w.vs] get] ; [set $w.bs] set $p3}
	3 {[set $w.rs] set $p1 ; [set $w.gs] set $p2 ; [set $w.bs] set [[set $w.vs] get]}
	4 {[set $w.rs] set $p3 ; [set $w.gs] set $p1 ; [set $w.bs] set [[set $w.vs] get]}
	5 {[set $w.rs] set [[set $w.vs] get] ; [set $w.gs] set $p1 ; [set $w.bs] set $p2}
	default {[set $w.rs] set 0 ; [set $w.gs] set 0 ; [set $w.bs] set 0}
    }

    global $w.col
    mesetcol [set $w.col] [[set $w.rs] get] [[set $w.gs] get] [[set $w.bs] get]

    update idletasks
}

proc mesetcol {col r g b} {
    set ir [expr int($r * 65535)]
    set ig [expr int($g * 65535)]
    set ib [expr int($b * 65535)]

    $col config -background [format #%04x%04x%04x $ir $ig $ib]
}

proc meset {w i} {
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

    global $w.rs $w.gs $w.bs

    global $color-r $color-g $color-b
    [set $w.rs] set [set $color-r]
    [set $w.gs] set [set $color-g]
    [set $w.bs] set [set $color-b]

    mesetrgb $w 0.0
}

proc mecommitcolor {w} {
    global $w.rs $w.gs $w.bs
    set r [[set $w.rs] get]
    set g [[set $w.gs] get]
    set b [[set $w.bs] get]
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

proc meresync {w} {
    global $w-ambient-r $w-ambient-g $w-ambient-b
    global $w-diffuse-r $w-diffuse-g $w-diffuse-b
    global $w-specular-r $w-specular-g $w-specular-b
    global $w-shininess
    global $w-emission-r $w-emission-g $w-emission-b
    global $w-reflectivity
    global $w-transparency
    global $w-refraction_index

    global $w.var
    setMaterial $w [set $w.var]

    set ir [expr int([set $w-ambient-r] * 65535)]
    set ig [expr int([set $w-ambient-g] * 65535)]
    set ib [expr int([set $w-ambient-b] * 65535)]
    $w.lmr.mr.right.amb config -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    set ir [expr int([set $w-diffuse-r] * 65535)]
    set ig [expr int([set $w-diffuse-g] * 65535)]
    set ib [expr int([set $w-diffuse-b] * 65535)]
    $w.lmr.mr.right.dif config -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    set ir [expr int([set $w-specular-r] * 65535)]
    set ig [expr int([set $w-specular-g] * 65535)]
    set ib [expr int([set $w-specular-b] * 65535)]
    $w.lmr.mr.right.spe config -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]
    set ir [expr int([set $w-emission-r] * 65535)]
    set ig [expr int([set $w-emission-g] * 65535)]
    set ib [expr int([set $w-emission-b] * 65535)]
    $w.lmr.mr.right.emi config -background [format #%04x%04x%04x $ir $ig $ib] \
	    -activebackground [format #%04x%04x%04x $ir $ig $ib]

    global $w-i
    meset $w [set $w-i]
}

proc mecommit {w command} {
    global $w $w.var
    setMaterial [set $w.var] $w
    eval $command
}

