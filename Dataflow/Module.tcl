
set port_spacing 18
set port_width 13
set port_height 7

itcl_class Module {
    constructor {config} {
    }
    method config {config} {}
    public name
    protected canvases ""
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_width 0
    public state "NeedData" {$this update_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}

    method set_state {st t} {
	set state $st
	set time $t
	update_state
	update_time
	update idletasks
    }
    method set_progress {p t} {
	set progress $p
	set time $t
	update_progress
	update_time
	update idletasks
	#tk_dialog .xx xx xx "" 0 OK
    }

    #
    #  Make the modules icon on a particular canvas
    #
    method make_icon {canvas modx mody} {
	lappend canvases $canvas
	set modframe $canvas.module$this
	frame $modframe -relief raised -borderwidth 3
	frame $modframe.ff
	pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
	set p $modframe.ff
	global ui_font
	global sci_root
	if {[$this info method ui] != ""} {
	    button $p.ui -text "UI" -borderwidth 2 -font $ui_font \
		    -anchor center -command "$this ui"
	    pack $p.ui -side left -ipadx 5 -ipady 2
	}
	global modname_font
	global time_font

	#
	# Make the title
	#
	label $p.title -text $name -font $modname_font -anchor w
	pack $p.title -side top -padx 2 -anchor w

	#
	# Make the time label
	#
	if {$make_time} {
	    label $p.time -text "00.00" -font $time_font
	    pack $p.time -side left -padx 2
	}

	#
	# Make the progress graph
	#
	if {$make_progress_graph} {
	    frame $p.inset -relief sunken -height 4 -borderwidth 2 \
		    -width $graph_width
	    pack $p.inset -side left -fill y -padx 2 -pady 2
	    frame $p.inset.graph -relief raised -width 0 -borderwidth 2 \
		    -background red
	    # Don't pack it in yet - the width is zero... 
	    #pack $p.inset.graph -fill y -expand yes -anchor nw
	}

	# Update the progress and time graphs
	update_progress
	update_time

	#
	# Stick it in the canvas
	#
	$canvas create window $modx $mody -window $modframe \
		-tags $this -anchor nw 

	#
	# Set up input/output ports
	#
	$this configureIPorts $canvas
	$this configureOPorts $canvas

	#
	# Try to find a position for the icon where it doesn't
	# overlap other icons
	#
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
		$canvas move $this 0 80
		incr mody 80
		if { $mody > [winfo height $canvas] } {
		    set mody 10
		    incr modx 200
		    $canvas coords $modx $mody
		}
	    }
	}
	
	menu $p.menu
	$p.menu add command -label "Help" -command "moduleHelp $name"
	$p.menu add command -label "Destroy" -command "moduleDestroy $canvas $this"

	bind $canvas <1> "$canvas raise current"
	bind $p <1> "moduleStartDrag $canvas $this %X %Y"
	bind $p <B1-Motion> "moduleDrag $canvas $this %X %Y"
	bind $p <ButtonRelease-1> "moduleEndDrag $modframe"
	bind $p <3> "tk_popup $p.menu %X %Y"
	bind $p.title <1> "moduleStartDrag $canvas $this %X %Y"
	bind $p.title <B1-Motion> "moduleDrag $canvas $this %X %Y"
	bind $p.title <ButtonRelease-1> "moduleEndDrag $modframe"
	bind $p.title <3> "tk_popup $p.menu %X %Y"
	if {$make_time} {
	    bind $p.time <1> "moduleStartDrag $canvas $this %X %Y"
	    bind $p.time <B1-Motion> "moduleDrag $canvas $this %X %Y"
	    bind $p.time <ButtonRelease-1> "moduleEndDrag $modframe"
	    bind $p.time <3> "tk_popup $p.menu %X %Y"
	}
	if {$make_progress_graph} {
	    bind $p.inset <1> "moduleStartDrag $canvas $this %X %Y"
	    bind $p.inset <B1-Motion> "moduleDrag $canvas $this %X %Y"
	    bind $p.inset <ButtonRelease-1> "moduleEndDrag $modframe"
	    bind $p.inset <3> "tk_popup $p.menu %X %Y"
	}
    }
    method configureAllIPorts {} {
	foreach t $canvases {
	    configureIPorts $t
	}
    }
    method configureAllOPorts {} {
	foreach t $canvases {
	    configureOPorts $t
	}
    }
    method configureIPorts {canvas} {
	set modframe $canvas.module$this
	set i 0
	while {[winfo exists $modframe.iport$i]} {
	    destroy $modframe.iport$i
	    destroy $modframe.iportlight$i
	    incr i
	}
	set portinfo [$this-c iportinfo]
	set i 0
	global port_spacing
	global port_width
	global port_height
	foreach t $portinfo {
	    set portcolor [lindex $t 0]
	    set connected [lindex $t 1]
	    set x [expr $i*$port_spacing+6]
	    if {$connected} {
		set e "outtop"
	    } else {
		set e "top"
	    }
	    bevel $modframe.iport$i -width $port_width \
		    -height $port_height -borderwidth 3 \
		    -edge $e -background $portcolor \
		    -pto 2 -pwidth 7 -pborder 2
	    place $modframe.iport$i -bordermode outside -x $x -y 0 -anchor nw
	    frame $modframe.iportlight$i -width $port_width -height 4 \
		    -relief raised -background black -borderwidth 0
	    place $modframe.iportlight$i -in $modframe.iport$i \
		    -x 0 -rely 1.0 -anchor nw
	    bind $modframe.iport$i <2> "startIPortConnection $this $i"
	    bind $modframe.iport$i <B2-Motion> \
		    "trackIPortConnection $this $i %x %y"
	    bind $modframe.iport$i <ButtonRelease-2> \
		    "endPortConnection \"$portcolor\""
	    incr i
	}
	rebuildConnections [netedit getconnected $this]
    }

    method configureOPorts {canvas} {
	set modframe $canvas.module$this

	set i 0
	while {[winfo exists $modframe.oport$i]} {
	    destroy $modframe.oport$i
	    destroy $modframe.oportlight$i
	    incr i
	}
	set portinfo [$this-c oportinfo]
	set i 0
	global port_spacing
	global port_width
	global port_height
	foreach t $portinfo {
	    set portcolor [lindex $t 0]
	    set connected [lindex $t 1]
	    set x [expr $i*$port_spacing+6]
	    if {$connected} {
		set e "outbottom"
	    } else {
		set e "bottom"
	    }
	    bevel $modframe.oport$i -width $port_width -height $port_height \
		    -borderwidth 3 -edge $e -background $portcolor \
		    -pto 2 -pwidth 7 -pborder 2
	    place $modframe.oport$i -bordermode ignore -rely 1 -anchor sw -x $x
	    frame $modframe.oportlight$i -width $port_width -height 4 \
		    -relief raised -background black -borderwidth 0
	    place $modframe.oportlight$i -in $modframe.oport$i \
		    -x 0 -y 0 -anchor sw
	    bind $modframe.oport$i <2> "startOPortConnection $this $i"
	    bind $modframe.oport$i <B2-Motion> \
		    "trackOPortConnection $this $i %x %y"
	    bind $modframe.oport$i <ButtonRelease-2> \
		"endPortConnection \"$portcolor\""
	    incr i
	}
	rebuildConnections [netedit getconnected $this]
    }
    method lightOPort {which color} {
	foreach t $canvases {
	    set p $t.module$this.oportlight$which
	    if {[winfo exists $p]} {
		$p configure -background $color
	    }
	}
    }
    method lightIPort {which color} {
	foreach t $canvases {
	    set p $t.module$this.iportlight$which
	    if {[winfo exists $p]} {
		$p configure -background $color
	    }
	}
    }
    method update_progress {} {
	if {!$make_progress_graph} return
	set width [expr int($progress*($graph_width-4))]
	if {$width == $old_width} return
	foreach t $canvases {
	    set modframe $t.module$this
	    if {$width == 0} {
		place forget $modframe.ff.inset.graph
	    } else {
		$modframe.ff.inset.graph configure -width $width
		if {$old_width == 0} {
		    place $modframe.ff.inset.graph -relheight 1 \
			    -anchor nw
		}
	    }
	}
	set old_width $width
    }
    method update_time {} {
	if {!$make_time} return
	if {$time < 60} {
	    set secs [expr int($time)]
	    set frac [expr int(100*($time-$secs))]
	    set tstr [format "%2d.%02d" $secs $frac]
	} elseif {$time < 3600} {
	    set mins [expr int($time/60)]
	    set secs [expr int($time-$mins*60)]
	    set tstr [format "%2d:%02d" $mins $secs]
	} else {
	    set hrs [expr int($time/3600)]
	    set mins [expr int($time-$hrs*3600)]
	    set tstr [format "%d::%02d" $hrs $mins]
	}
	foreach t $canvases {
	    set modframe $t.module$this
	    $modframe.ff.time configure -text $tstr
	}
    }
    method update_state {} {
	if {!$make_progress_graph} return
	if {$state == "Executing"} {
	    set p 0
	    set color red
	} elseif {$state == "Completed"} {
	    set p 1
	    set color green
	} elseif {
	    set width 0
	    set p 0
	}
	foreach t $canvases {
	    set modframe $t.module$this
	    $modframe.ff.inset.graph configure -background $color
	}
	#
	# call update_progress
	#
	set progress $p
	update_progress
    }
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
    } else {
	set mx $ox
	if {$ix < $mx} {
	    set mx $ix
	}
	return [list $ox $oy $ox [expr $oy+10] [expr $mx-50] [expr $oy+10] \
		[expr $mx-50] [expr $iy-10] $ix [expr $iy-10] $ix $iy]
    }
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

proc configureIPorts {modid} {
    if {[info command $modid] != ""} {
	$modid configureAllIPorts
    }
}

proc configureOPorts {modid} {
    if {[info command $modid] != ""} {
	$modid configureAllOPorts
    }
}

proc moduleHelp {name} {
    global sci_root
    helpPage [glob $sci_root/Modules/*/help/$name.html]
}

proc moduleDestroy {c modid} {
    puts "moduleDestroy not implemented."
}

