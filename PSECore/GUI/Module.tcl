
set port_spacing 18
set port_width 13
set port_height 7

itcl_class Module {
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
			
    constructor {config} {

        global $this-notes
	if [info exists $this-notes] {
	    set dum 0
	} else {set $this-notes ""}
    }

    destructor {
	puts "Module Base Class Destructor Called"
    }

    method config {config} {
    }

    public name
    protected canvases ""
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_width 0
    public state "NeedData" {$this update_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}

    public group    -1
    public selected  0

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
    method make_icon {canvas minicanvas modx mody} {

	global mainCanvasWidth mainCanvasHeight
	#set modx [expr int([expr (([lindex [$canvas xview] 0]*$mainCanvasWidth)+$modx)])]
	#set mody [expr int([expr (([lindex [$canvas yview] 0]*$mainCanvasHeight)+$mody)])]
	
	lappend canvases $canvas
	set modframe $canvas.module[modname]
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

	#  Make the mini module icon on a particular canvas

	set miniframe $minicanvas.module[modname]

	frame $miniframe -borderwidth 0
	frame $miniframe.ff
	pack $miniframe.ff -side top -expand yes \
		-fill both -padx 2 -pady 1

	global SCALEX SCALEY
	global basecolor
	$minicanvas create rectangle \
		[expr $modx/$SCALEX] [expr $mody/$SCALEY] \
		[expr $modx/$SCALEX + 4] [expr $mody/$SCALEY + 2] \
		-outline "" -fill $basecolor \
		-tags [modname]

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
		    -background green
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
		-tags [modname] -anchor nw 

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

	    if { $l == 0 || $l == 1 || $l == 2 } {
		set done 1
	    } else {
		$canvas move [modname] 0 80
		$minicanvas move [modname] 0 [expr 80 / $SCALEY ]
		incr mody 80


		set canbot [expr int( \
		       [lindex [$canvas yview] 0]*$mainCanvasHeight + \
		       [winfo height $canvas] ) ]

		if { $mody > $canbot } {
		    set mody [expr $mody - [winfo height $canvas]]
		    incr modx 200
		    
		    $canvas coords [modname] $modx $mody
		    $minicanvas coords [modname] [expr $modx / $SCALEX] [expr $mody / $SCALEY] [expr ($modx+120) / $SCALEX] [expr ($mody+50) /$SCALEY]
		}
	    }
	}
	                            
	menu $p.menu -tearoff false -disabledforeground white
	$p.menu add command -label "$this" -state disabled
	$p.menu add separator
	$p.menu add command -label "Execute" -command "$this-c needexecute"
	$p.menu add command -label "Help" -command "moduleHelp $name"

# This menu item was added by Mohamed Dekhil for the CSAFE project
	$p.menu add command -label "Notes" -command "moduleNotes $name [modname]"

	$p.menu add command -label "Destroy" \
		-command "moduleDestroy $canvas $minicanvas [modname]"


	bind $p <2> "$this toggleSelected"

	bind $p <1> "$canvas raise $this"
	bind $p <1> "moduleStartDrag $canvas [modname] %X %Y"
	bind $p <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $p <ButtonRelease-1> "moduleEndDrag $modframe"
	bind $p <3> "tk_popup $p.menu %X %Y"
	bind $p.title <1> "moduleStartDrag $canvas [modname] %X %Y"
	bind $p.title <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $p.title <ButtonRelease-1> "moduleEndDrag $modframe"
	bind $p.title <3> "tk_popup $p.menu %X %Y"
	if {$make_time} {
	    bind $p.time <1> "moduleStartDrag $canvas [modname] %X %Y"
	    bind $p.time <B1-Motion> \
		    "moduleDrag $canvas $minicanvas [modname] %X %Y"
	    bind $p.time <ButtonRelease-1> "moduleEndDrag $modframe"
	    bind $p.time <3> "tk_popup $p.menu %X %Y"
	}
	if {$make_progress_graph} {
	    bind $p.inset <1> "moduleStartDrag $canvas [modname] %X %Y"
	    bind $p.inset <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
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
	set modframe $canvas.module[modname]
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
	    bind $modframe.iport$i <2> "startIPortConnection [modname] $i %x %y"
	    bind $modframe.iport$i <B2-Motion> \
		    "trackIPortConnection [modname] $i %x %y"
	    bind $modframe.iport$i <ButtonRelease-2> \
		    "endPortConnection \"$portcolor\""
	    incr i
	}
	rebuildConnections [netedit getconnected [modname]]
    }

    method configureOPorts {canvas} {
	set modframe $canvas.module[modname]

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
	    bind $modframe.oport$i <2> "startOPortConnection [modname] $i %x %y"
	    bind $modframe.oport$i <B2-Motion> \
		    "trackOPortConnection [modname] $i %x %y"
	    bind $modframe.oport$i <ButtonRelease-2> \
		"endPortConnection \"$portcolor\""
	    incr i
	}
	rebuildConnections [netedit getconnected [modname]]
    }

    method toggleSelected {} {

	puts "toggle Selected"
	
	if { $selected == 0 } {
	    set selected 1
	    .top.globalViewFrame.canvas configure -bg red
	} else {
	    set selected 0
	    .top.globalViewFrame.canvas configure -bg blue
	}

    }


    method lightOPort {which color} {
	foreach t $canvases {
	    set p $t.module[modname].oportlight$which
	    if {[winfo exists $p]} {
		$p configure -background $color
	    }
	}
    }
    method lightIPort {which color} {
	foreach t $canvases {
	    set p $t.module[modname].iportlight$which
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
	    set modframe $t.module[modname]
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
	#puts "state is $state"
	if {$state == "JustStarted"} {
	    set tstr " ?.??"
	} else {
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
	}
	foreach t $canvases {
	    set modframe $t.module[modname]
	    $modframe.ff.time configure -text $tstr
	}
    }

    method update_state {} {
	if {!$make_progress_graph} return
	if {$state == "JustStarted 1123"} {
	    set p 0.5
	    set color red
	} elseif {$state == "Executing"} {
	    set p 0
	    set color red
	} elseif {$state == "NeedData"} {
	    set p 1
	    set color yellow
	} elseif {$state == "Completed"} {
	    set p 1
	    set color green
	} else {
	    set width 0
		set color grey75
	    set p 0
	}
	foreach t $canvases {
	    set modframe $t.module[modname]
	    $modframe.ff.inset.graph configure -background $color
	}
	#
	# call update_progress
	#
	set progress $p
	update_progress
    }
    method get_x {} {
	set canvas [lindex $canvases 0]
	set coords [$canvas coords [modname]]
	return [lindex $coords 0]
    }
    method get_y {} {
	set canvas [lindex $canvases 0]
	set coords [$canvas coords [modname]]
	return [lindex $coords 1]
    }
}

proc startIPortConnection {imodid iwhich x y} {
    # Find all of the OPorts of the same type and draw a temporary line
    # to them....
    global conn_oports
    set conn_oports [netedit findoports $imodid $iwhich]
    global netedit_canvas
    set coords [computeIPortCoords $imodid $iwhich]
    set typename [lindex [lindex [$imodid-c iportinfo] $iwhich] 2]
    set portname [lindex [lindex [$imodid-c iportinfo] $iwhich] 3]
    set fullname $typename:$portname
    eval $netedit_canvas create text [lindex $coords 0] [lindex $coords 1] \
	    -anchor sw -text {$fullname} -tags "tempname"
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

proc startOPortConnection {omodid owhich x y} {
    # Find all of the IPorts of the same type and draw a temporary line
    # to them....
    global conn_iports
    set conn_iports [netedit findiports $omodid $owhich]
    global netedit_canvas
    set coords [computeOPortCoords $omodid $owhich]
    set typename [lindex [lindex [$omodid-c oportinfo] $owhich] 2]
    set portname [lindex [lindex [$omodid-c oportinfo] $owhich] 3]
    set fullname $typename:$portname
    eval $netedit_canvas create text [lindex $coords 0] [lindex $coords 1] \
	    -anchor nw -text {$fullname} -tags "tempname"
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

    set minipath ""

    global SCALEX SCALEY
    set doingX 1
    foreach point $path {
	if [expr $doingX ] {
	    lappend minipath [expr $point/$SCALEX] 
	} else {
	    lappend minipath [expr $point/$SCALEY] 
	}
	set doingX [expr !$doingX]
    }

    global netedit_canvas netedit_mini_canvas
    eval $netedit_canvas create bline $path -width 7 \
	-borderwidth 2 -fill \"$portcolor\" \
	-tags $connid
    $netedit_canvas bind $connid <ButtonPress-3> \
	 "destroyConnection $connid $omodid $imodid"

    eval $netedit_mini_canvas create line $minipath -width 1 \
	-fill \"$portcolor\" -tags $connid

    $netedit_mini_canvas lower $connid
}

proc destroyConnection {connid omodid imodid} {
    netedit deleteconnection $connid $omodid $imodid
    global netedit_canvas netedit_mini_canvas
    $netedit_canvas delete $connid
    $netedit_mini_canvas delete $connid
    configureOPorts $omodid
    configureIPorts $imodid
}
	
proc rebuildConnection {connid omodid owhich imodid iwhich} {
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    global netedit_canvas netedit_mini_canvas

    eval $netedit_canvas coords $connid $path

    set minipath ""
    global SCALEX SCALEY

    set doingX 1
    foreach point $path {
	if [expr $doingX ] {
	    lappend minipath [expr round($point/$SCALEX)] 
	} else {
	    lappend minipath [expr round($point/$SCALEY)] 
	}
	set doingX [expr !$doingX]
    }
    eval $netedit_mini_canvas coords $connid $minipath
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


    set relativeMouseX [expr $x+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*4500))])]
    set relativeMouseY [expr $y+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*4500))])]

    set mindist [computeDist $relativeMouseX $relativeMouseY $ix $iy]
    set mindist 6364
    set minport ""
    global conn_iports

    foreach i $conn_iports {
	set imodid [lindex $i 0]
	set iwhich [lindex $i 1]

	set c [computeIPortCoords $imodid $iwhich]
	set ox [lindex $c 0]
	set oy [lindex $c 1]

	set dist [computeDist $relativeMouseX $relativeMouseY $ox $oy]
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
    $netedit_canvas delete tempname
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


proc moduleDrag {maincanvas minicanvas modid x y} {
    
    global xminwarped
    global xmaxwarped
    global yminwarped
    global ymaxwarped
    global lastX lastY
    global SCALEX SCALEY

    
    # Canvas-relative X and Y module coordinates
    set modxpos [ lindex [ $maincanvas coords $modid ] 0 ]
    set modypos [ lindex [ $maincanvas coords $modid ] 1 ]
    
    # X and Y coordinates of canvas origin
    set Xbounds [ winfo rootx $maincanvas ]
    set Ybounds [ winfo rooty $maincanvas ]
    
    # Canvas width and height
    set canwidth [ winfo width $maincanvas ]
    set canheight [winfo height $maincanvas ]
    
    # Canvas-relative max module bounds coordinates
    set mmodxpos [ lindex [$maincanvas bbox $modid ] 2]
    set mmodypos [ lindex [$maincanvas bbox $modid ] 3]
    
    # Absolute max canvas coordinates
    set maxx [expr $Xbounds+$canwidth]
    set maxy [expr $Ybounds+$canheight]
    
    # Absolute canvas max coordinates 
    set ammodxpos [expr $Xbounds+$mmodxpos]
    set ammodypos [expr $Ybounds+$mmodypos]
    
    global mainCanvasWidth mainCanvasHeight
    
    # Current canvas relative minimum viewable-canvas bounds
    set currminxbdr [expr ([lindex [$maincanvas xview] 0]*$mainCanvasWidth)]
    set currminybdr [expr ([lindex [$maincanvas yview] 0]*$mainCanvasHeight)]
    
    # Current canvas relative maximum viewable-canvas bounds
    set currxbdr [expr $canwidth + ([lindex [$maincanvas xview] 0]*$mainCanvasWidth)]
    set currybdr [expr $canheight + ([lindex [$maincanvas yview] 0]*$mainCanvasHeight)]

    # Cursor warping flags
    set xminwarped 0
    set xmaxwarped 0
    set yminwarped 0
    set ymaxwarped 0

    set xs 0
    set ys 0
    
    set currx [expr $x-$Xbounds]

    set mainCanvasWidth [expr double($mainCanvasWidth)]
    set mainCanvasHeight [expr double($mainCanvasHeight)]
    #############################################
    
    # if user attempts to drag module off near end of canvas
    
    if { [expr $modxpos+($x-$lastX)] <= $currminxbdr} {
	
	

	#if viewable canvas is not on the border of the main canvas

	if { $currminxbdr > 0} {
	    set xbegView [lindex [$maincanvas xview] 0]
	    set xdiff [expr ($modxpos+($x-$lastX))-$currminxbdr]
	    set mvx [expr (($xdiff/$mainCanvasWidth)+$xbegView)]
	    $maincanvas xview moveto $mvx
	}
    
	#if viewable canvas is on the border of the main canvas

	if { [expr $modxpos+($x-$lastX)] <= 0 } {
	    $maincanvas move $modid [expr -$modxpos] 0
	    $minicanvas move $modid [expr (-$modxpos)/$SCALEX] 0
	    set lastX $x
	}

}
    
    #if user attempts to drag module off far end of canvas

    if { [expr $mmodxpos+($x-$lastX)] >= $currxbdr} {
		
	if {$currxbdr < $mainCanvasWidth} {
	    
	    puts "here"
	    #if not on edge of canvas, move viewable area right	 
	    set xbegView [lindex [$maincanvas xview] 0]
	    set xdiff [expr ($mmodxpos+($x-$lastX))-$currxbdr]
	    puts "xbegView: $xbegView"
	    puts "$mainCanvasWidth"
	    set mvx [expr (($xdiff/$mainCanvasWidth)+$xbegView)]
	    puts "mvx: $mvx"
	    $maincanvas xview moveto $mvx
	}
	
	# if the right side of the module is at the right edge
	# of the canvas.

	if { [expr $mmodxpos+($x-$lastX)] >= $mainCanvasWidth} {

	    # dont' let the module move off the right side of the
	    # entire canvas
	    
	    $maincanvas move $modid [expr ($mainCanvasWidth-$mmodxpos)] 0
	    $minicanvas move $modid [expr (($mainCanvasWidth-$mmodxpos)/$SCALEX)] 0
	    set lastX $x
	}

    }
    
    #cursor-boundary check and warp for x-axis

    if { [expr $x-$Xbounds] > $canwidth } {
	cursor warp $maincanvas $canwidth [expr $y-$Ybounds]
	set currx $canwidth
	set xmaxwarped 1
    }
    
    if { [expr $x-$Xbounds] < 0 } {
	cursor warp $maincanvas 0 [expr $y-$Ybounds]
	set currx 0
	set xminwarped 1
	
    }
    
#Y boundary checks

    if { [expr $modypos+($y-$lastY)] <= $currminybdr} {
	if {$currminybdr > 0} {
	    set ybegView [lindex [$maincanvas yview] 0]
	    set ydiff [expr ($modypos+($y-$lastY))-$currminybdr]
	    set mvy [expr (($ydiff/$mainCanvasHeight)+$ybegView)]
	    $maincanvas yview moveto $mvy
	}
    
	#if viewable canvas is on the border of the main canvas

	if { [expr $modypos+($y-$lastY)] <= 0 } {
	    $maincanvas move $modid 0 [expr -$modypos]
	    $minicanvas move $modid 0 [expr (-$modypos)/$SCALEY]
	    set lastY $y
	}

    }
 
#if user attempts to drag module off far end of canvas
    
#round currybdr
    set currybdr [expr int($currybdr+.5)]

if { [expr $mmodypos+($y-$lastY)] >= $currybdr} {
		
	if {$currybdr < $mainCanvasHeight} {
	    #if not on edge of canvas, move viewable area down

	    set ybegView [lindex [$maincanvas yview] 0]
	    set ydiff [expr ($mmodypos+($y-$lastY))-$currybdr]
	    set mvy [expr (($ydiff/$mainCanvasHeight)+$ybegView)]
	    $maincanvas yview moveto $mvy
	}
	
	# if the bottom side of the module is at the bottom edge
	# of the canvas.

	if { [expr $mmodypos+($y-$lastY)] >= $mainCanvasHeight} {

	    # dont' let the module move off the bottom side of the
	    # entire canvas
	    
	    $maincanvas move $modid 0 [expr ($mainCanvasHeight-$mmodypos)]
	    $minicanvas move $modid 0 [expr (($mainCanvasHeight-$mmodypos)/$SCALEY)]
	    set lastY $y
	}

    }



#cursor-boundary check and warp for y-axis

    if { [expr $y-$Ybounds] < 0 } {
	cursor warp $maincanvas $currx 0
	set yminwarped 1
    }
    
    if { [expr $y-$Ybounds] > $canheight } {
	cursor warp $maincanvas $currx $canheight
	set ymaxwarped 1
    }

    
  #####################################################################
    
    $maincanvas move $modid [expr $x-$lastX] [expr $y-$lastY]
    
    $minicanvas move $modid [expr ( $x - $lastX ) / $SCALEX ] \
	                    [expr ( $y - $lastY ) / $SCALEY ]
    
    #if the mouse has been warped, adjust $lastX accordingly
    
 

    if { $xmaxwarped } {
	set lastX [expr $maxx - [.bot.neteditFrame.vscroll cget -width] - 5]
	set xs 1
    } 
    if { $xminwarped } {
	set lastX $Xbounds
	set xs 1
    } 
    if { $yminwarped } {
	set lastY $Ybounds
	set ys 1
    } 
    if { $ymaxwarped } {
	set lastY [expr $maxy - [.bot.neteditFrame.hscroll cget -width] - 5]
	set ys 1
    } 
    if { $xs==0 } {
	set lastX $x
    }
    if { $ys==0 } {
	set lastY $y
    }

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
    global pse_root
    tk_messageBox -message "For help on this module, point your web browser at:\n$pse_root/GuiFiles/help/$name" 
#    helpPage [glob $pse_root/help/$name.html]
}

# By Mohamed Dekhil

proc moduleNotes {name mclass} {
    
    global $mclass-notes
    set w .module_notes
    toplevel $w
    text $w.tnotes -relief sunken -bd 2 
    frame $w.fbuttons 
    button $w.fbuttons.ok -text "Done" -command "okNotes $w $mclass"
    button $w.fbuttons.cancel -text "Cancel" -command "destroy $w"
    
    pack $w.tnotes $w.fbuttons -side top -padx 5 -pady 5
    pack $w.fbuttons.ok -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
    if [info exists $mclass-notes] {$w.tnotes insert 1.0 [set $mclass-notes]}
}

# By Mohamed Dekhil

proc okNotes {w mclass} {
    global $mclass-notes
    set  $mclass-notes [$w.tnotes get 1.0 end]
    destroy $w
}


proc moduleDestroy {maincanvas minicanvas modid} {
    set modList [netedit getconnected $modid]
    foreach i $modList {
	set connid [lindex $i 0]
	set omodid [lindex $i 1]
	set owhich [lindex $i 2]
	set imodid [lindex $i 3]
	set iwhich [lindex $i 4]
	destroyConnection $connid $omodid $imodid
    }

    $maincanvas delete $modid
    destroy ${maincanvas}.module$modid
    $minicanvas delete $modid
    destroy $minicanvas.module$modid
    netedit deletemodule $modid
    $modid delete
   
    if {[winfo exists .ui$modid]} {
	destroy .ui$modid
    }

}
