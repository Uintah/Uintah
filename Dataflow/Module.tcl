
proc makeModule {modid name canvas} {
    set modx 10
    set mody 10
    set modframe $canvas.module$modid
    global moduleframe
    set moduleframe($modid) $modframe
    frame $modframe -relief raised -borderwidth 3
    frame $modframe.ff
    pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
    set p $modframe.ff
    global ui_font
    global sci_root
    set script_name [glob -nocomplain $sci_root/Modules/*/$name.tcl]
    if {$script_name != ""} {
	button $p.ui -text "UI" -borderwidth 2 -font $ui_font -anchor center \
		-command "source $script_name ; source $sci_root/TCL/Filebox.tcl ; ui$name $modid"
	pack $p.ui -side left -ipadx 5 -ipady 2
    }
    global modname_font
    global time_font
    label $p.title -text $name -font $modname_font -anchor w
    pack $p.title -side top -padx 2 -anchor w
    label $p.time -text "0.00" -font $time_font
    pack $p.time -side left -padx 2
    frame $p.inset -relief sunken -height 4 -borderwidth 2 \
	-width 40
    pack $p.inset -side left -expand yes -fill both -padx 2 -pady 2
    frame $p.inset.graph -relief raised -width 0 -borderwidth 2 \
	-background red
    pack $p.inset.graph -fill y -expand yes -anchor nw
    $canvas create window $modx $mody -window $modframe \
	    -tags $modid -anchor nw 

    configureIPorts $modid
    configureOPorts $modid

    set done 0	
    while { $done == 0 } {
	set x1 $modx 
	set y1 $mody
	set x2 [expr $modx+120]
	set y2 [expr $mody+50]

	set l [llength [$canvas find overlapping $x1 $y1 $x2 $y2]]
	if { $l == 0 || $l == 1 } {
	    set done 1
	} else {
	    $canvas move $modid 0 80
	    incr mody 80
	    if { $mody > [winfo height $canvas] } {
		set mody 10
		incr modx 200
		$canvas coords $modx $mody
	    }
	}
    }

    bind $canvas <1> "$canvas raise current"
    bind $p <1> "moduleStartDrag $canvas $modid %X %Y"
    bind $p <B1-Motion> "moduleDrag $canvas $modid %X %Y"
    bind $p <ButtonRelease-1> "moduleEndDrag $modframe"
    bind $p.title <1> "moduleStartDrag $canvas $modid %X %Y"
    bind $p.title <B1-Motion> "moduleDrag $canvas $modid %X %Y"
    bind $p.title <ButtonRelease-1> "moduleEndDrag $modframe"
    bind $p.time <1> "moduleStartDrag $canvas $modid %X %Y"
    bind $p.time <B1-Motion> "moduleDrag $canvas $modid %X %Y"
    bind $p.time <ButtonRelease-1> "moduleEndDrag $modframe"
    bind $p.inset <1> "moduleStartDrag $canvas $modid %X %Y"
    bind $p.inset <B1-Motion> "moduleDrag $canvas $modid %X %Y"
    bind $p.inset <ButtonRelease-1> "moduleEndDrag $modframe"
}
set port_spacing 18
set port_width 13
set port_height 7

proc configureIPorts {modid} {
    global moduleframe
    if {![info exists moduleframe($modid)]} {
	return;
    }
    set w $moduleframe($modid)
    global port_spacing
    global port_width
    global port_height

    set i 0
    while {[winfo exists $w.iport$i]} {
	destroy $w.iport$i
	destroy $w.iportlight$i
	incr i
    }
    set portinfo [$modid iportinfo]
    set i 0
    foreach t $portinfo {
	set portcolor [lindex $t 0]
	set connected [lindex $t 1]
	set x [expr $i*$port_spacing+6]
	if {$connected} {
	    set e "outtop"
	} else {
	    set e "top"
	}
	bevel $w.iport$i -width $port_width -height $port_height -borderwidth 3 \
		-edge $e -background $portcolor \
		-pto 2 -pwidth 7 -pborder 2
	place $w.iport$i -bordermode outside -x $x -y 0 -anchor nw
	frame $w.iportlight$i -width $port_width -height 4 \
		-relief raised -background black -borderwidth 0
	place $w.iportlight$i -in $w.iport$i \
		-x 0 -rely 1.0 -anchor nw
	bind $w.iport$i <2> "startIPortConnection $modid $i"
	bind $w.iport$i <B2-Motion> "trackIPortConnection $modid $i %x %y"
	bind $w.iport$i <ButtonRelease-2> "endPortConnection \"$portcolor\""
	incr i
    }
    rebuildConnections [netedit getconnected $modid]
}

proc configureOPorts {modid} {
    global moduleframe
    if {![info exists moduleframe($modid)]} {
	return;
    }
    set w $moduleframe($modid)
    global port_width
    global port_height
    global port_spacing

    set i 0
    while {[winfo exists $w.oport$i]} {
	destroy $w.oport$i
	destroy $w.oportlight$i
	incr i
    }
    set portinfo [$modid oportinfo]
    set i 0
    foreach t $portinfo {
	set portcolor [lindex $t 0]
	set connected [lindex $t 1]
	set x [expr $i*$port_spacing+6]
	if {$connected} {
	    set e "outbottom"
	} else {
	    set e "bottom"
	}
	bevel $w.oport$i -width $port_width -height $port_height \
	    -borderwidth 3 -edge $e -background $portcolor \
	    -pto 2 -pwidth 7 -pborder 2
	place $w.oport$i -bordermode ignore -rely 1 -anchor sw -x $x
	frame $w.oportlight$i -width $port_width -height 4 \
	    -relief raised -background black -borderwidth 0
	place $w.oportlight$i -in $w.oport$i \
	    -x 0 -y 0 -anchor sw
	bind $w.oport$i <2> "startOPortConnection $modid $i"
	bind $w.oport$i <B2-Motion> "trackOPortConnection $modid $i %x %y"
	bind $w.oport$i <ButtonRelease-2> "endPortConnection \"$portcolor\""
	incr i
    }
    rebuildConnections [netedit getconnected $modid]
}

proc startIPortConnection {imodid iwhich} {
    # Find all of the OPorts of the same type and draw a temporary line
    # to them....
    global conn_oports
    set conn_oports [netedit findoports $imodid $iwhich]
    global netedit_canvas
    foreach i $conn_oports {
	set omodid [lindex $i 0]
	set owhich [lindex $i 1]
	set path [join [routeConnection $omodid $owhich $imodid $iwhich]]
	eval $netedit_canvas create line $path -width 2 \
	    -tags \"tempconnections iconn$owhich$omodid\"
    }
    global potential_connection
    set potential_connection ""
}

proc startOPortConnection {omodid owhich} {
    # Find all of the IPorts of the same type and draw a temporary line
    # to them....
    global conn_iports
    set conn_iports [netedit findiports $omodid $owhich]
    global netedit_canvas
    foreach i $conn_iports {
	set imodid [lindex $i 0]
	set iwhich [lindex $i 1]
	set path [join [routeConnection $omodid $owhich $imodid $iwhich]]
	eval $netedit_canvas create line $path -width 2 \
	    -tags \"tempconnections oconn$iwhich$imodid\"
    }
    global potential_connection
    set potential_connection ""
}

proc buildConnection {connid portcolor omodid owhich imodid iwhich} {
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    global netedit_canvas
    eval $netedit_canvas create bline $path -width 7 \
	-borderwidth 2 -fill \"$portcolor\" \
	-tags $connid
}

proc rebuildConnection {connid omodid owhich imodid iwhich} {
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    global netedit_canvas
    eval $netedit_canvas coords $connid $path
}

proc rebuildConnections {list} {
    foreach i $list {
	set connid [lindex $i 0]
	set omodid [lindex $i 1]
	set owhich [lindex $i 2]
	set imodid [lindex $i 3]
	set iwhich [lindex $i 4]
	rebuildConnection $connid $omodid $owhich $imodid $iwhich
    }
}

proc trackIPortConnection {imodid which x y} {
    # Get coords in canvas
    global netedit_canvas
    set ox1 [winfo x $netedit_canvas.module$imodid.iport$which]
    set ox2 [winfo x $netedit_canvas.module$imodid]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $netedit_canvas.module$imodid.iport$which]
    set oy2 [winfo y $netedit_canvas.module$imodid]
    set y [expr $y+$oy1+$oy2]

    set c [computeIPortCoords $imodid $which]
    set ix [lindex $c 0]
    set iy [lindex $c 1]
    set mindist [computeDist $x $y $ix $iy]
    set minport ""
    global conn_oports
    foreach i $conn_oports {
	set omodid [lindex $i 0]
	set owhich [lindex $i 1]
	set c [computeOPortCoords $omodid $owhich]
	set ox [lindex $c 0]
	set oy [lindex $c 1]
	set dist [computeDist $x $y $ox $oy]
	if {$dist < $mindist} {
	    set mindist $dist
	    set minport $i
	}
    }
    $netedit_canvas itemconfigure tempconnections -fill black
    global potential_connection
    if {$minport != ""} {
	set omodid [lindex $minport 0]
	set owhich [lindex $minport 1]
	$netedit_canvas itemconfigure iconn$owhich$omodid -fill red
	set potential_connection [list $omodid $owhich $imodid $which]
    } else {
	set potential_connection ""
    }
}

proc trackOPortConnection {omodid which x y} {
    # Get coords in canvas
    global netedit_canvas
    set ox1 [winfo x $netedit_canvas.module$omodid.oport$which]
    set ox2 [winfo x $netedit_canvas.module$omodid]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $netedit_canvas.module$omodid.oport$which]
    set oy2 [winfo y $netedit_canvas.module$omodid]
    set y [expr $y+$oy1+$oy2]

    set c [computeOPortCoords $omodid $which]
    set ix [lindex $c 0]
    set iy [lindex $c 1]
    set mindist [computeDist $x $y $ix $iy]
    set minport ""
    global conn_iports
    foreach i $conn_iports {
	set imodid [lindex $i 0]
	set iwhich [lindex $i 1]
	set c [computeIPortCoords $imodid $iwhich]
	set ox [lindex $c 0]
	set oy [lindex $c 1]
	set dist [computeDist $x $y $ox $oy]
	if {$dist < $mindist} {
	    set mindist $dist
	    set minport $i
	}
    }
    $netedit_canvas itemconfigure tempconnections -fill black
    global potential_connection
    if {$minport != ""} {
	set imodid [lindex $minport 0]
	set iwhich [lindex $minport 1]
	$netedit_canvas itemconfigure oconn$iwhich$imodid -fill red
	set potential_connection [list $omodid $which $imodid $iwhich]
    } else {
	set potential_connection ""
    }
}

proc endPortConnection {portcolor} {
    global netedit_canvas
    $netedit_canvas delete tempconnections
    global potential_connection
    if {$potential_connection != ""} {
	set omodid [lindex $potential_connection 0]
	set owhich [lindex $potential_connection 1]
	set imodid [lindex $potential_connection 2]
	set iwhich [lindex $potential_connection 3]
	set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
	buildConnection $connid $portcolor $omodid $owhich $imodid $iwhich
	configureOPorts $omodid
	configureIPorts $imodid
    }
}

proc routeConnection {omodid owhich imodid iwhich} {
    set outpos [computeOPortCoords $omodid $owhich]
    set inpos [computeIPortCoords $imodid $iwhich]
    set ox [lindex $outpos 0]
    set oy [lindex $outpos 1]
    set ix [lindex $inpos 0]
    set iy [lindex $inpos 1]

    set minextend 10
    if {$ox == $ix} {
	return [list $ox $oy $ix $iy]
    } elseif {[expr $oy+2*$minextend] < $iy} {
	set my [expr ($oy+$iy)/2]
	return [list $ox $oy $ox $my $ix $my $ix $iy]
    }
    return "oops"
}

proc computeIPortCoords {modid which} {
    global netedit_canvas
    global port_spacing
    global port_width
    set px [expr $which*$port_spacing+6+$port_width/2]
    set at [$netedit_canvas coords $modid]
    set mx [lindex $at 0]
    set my [lindex $at 1]
    set x [expr $px+$mx]
    set y $my
    return [list $x $y]
}

proc computeOPortCoords {modid which} {
    global netedit_canvas
    global port_spacing
    global port_width
    set px [expr $which*$port_spacing+6+$port_width/2]
    set at [$netedit_canvas coords $modid]
    set mx [lindex $at 0]
    set my [lindex $at 1]
    set h [winfo height $netedit_canvas.module$modid]
    set x [expr $px+$mx]
    set y [expr $my+$h]
    return [list $x $y]
}

proc computeDist {x1 y1 x2 y2} {
    set dx [expr $x2-$x1]
    set dy [expr $y2-$y1]
    return [expr sqrt($dx*$dx+$dy*$dy)]
}

proc moduleStartDrag {c modid x y} {
    global lastX lastY
    set lastX $x
    set lastY $y
    global moduleDragged
    set moduleDragged 0
    global moduleConnected
    set moduleConnected [netedit getconnected $modid]
}

proc moduleDrag {c modid x y} {
    global lastX lastY
    $c move $modid [expr $x-$lastX] [expr $y-$lastY]
    set lastX $x
    set lastY $y
    global moduleDragged
    set moduleDragged 1
    global moduleConnected
    rebuildConnections $moduleConnected
}

proc moduleEndDrag {mframe} {
    global moduleDragged
    if {$moduleDragged == 0} {
	raise $mframe
    }
}

proc updateProgress {modid p time} {
    global moduleframe
    set w $moduleframe($modid)
    set width [winfo width $w.ff.inset]
    $w.ff.inset.graph configure -width [expr $p*($width-4)]
    updateTime $modid $time
}

proc updateState {modid state time} {
    global moduleframe
    set w $moduleframe($modid)
    if {$state == "Executing"} {
	set width 0
	set color red
    } elseif {$state == "Completed"} {
	set width [expr [winfo width $w.ff.inset]-4]
	set color green
    } elseif {
	set width 0
	set color black
    }
    $w.ff.inset.graph configure -width $width -background $color
    updateTime $modid $time
}

proc updateTime {modid time} {
    global moduleframe
    set w $moduleframe($modid)
    $w.ff.time configure -text $time
    update idletasks
}
