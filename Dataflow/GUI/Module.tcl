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
global port_spacing
set port_spacing 18

global port_width
set port_width 13

global port_height
set port_height 7 

global selected_color
#set selected_color blue
set selected_color darkgray

global unselected_color
set unselected_color gray

global CurrentlySelectedModules
set CurrentlySelectedModules ""

global modules
set modules ""

global undoList
set undoList ""

global redoList
set redoList ""


itcl_class Module {
   
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
			
    constructor {config} {
	set msgLogStream [TclStream msgLogStream#auto]

	global $this-notes	
	if ![info exists $this-notes] {
	    set $this-notes ""
	}
	
	# messages should be accumulating
	if {[info exists $this-msgStream]} {
	    $msgLogStream registerVar $this-msgStream
	}
    }
    
    destructor {
	set w .mLogWnd[modname]
	if [winfo exists $w] {
	    destroy $w
	}
	$msgLogStream destructor
	eval unset [info vars $this-*]
	destroy $this
    }
    
    public msgLogStream
    public name
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_width 0
    protected indicator_width 15
    protected initial_width 0
    protected mconnected {}
    protected made_icon 0
    public state "NeedData" {$this update_state}
    public msg_state "Reset" {$this update_msg_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}

    method name {} {
	return $name
    }
    
    method set_state {st t} {
	set state $st
	set time $t
	update_state
	update_time
	update idletasks
    }

    method set_msg_state {st} {
	set msg_state $st
	update_msg_state
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

    method set_title {name} {
	# index points to the second "_" in the name, after package & category
	set index [string first "_" $name [expr [string first "_" $name 0]+1]]
	return [string range $name [expr $index+1] end]
    }

    method initialize_ui { {my_display "local"} } {
        $this ui
	if {[winfo exists .ui[modname]]!= 0} {
	    set w .ui[modname]
	    wm title $w [set_title [modname]]
	}
    }

    method have_ui {} {
	if {[$this info method ui] != ""} {
	    return 1;
	} else {
	    return 0;
	}
    }

    #  Make the modules icon on a particular canvas
    method make_icon {canvas minicanvas modx mody} {
	global $this-done_bld_icon
	set $this-done_bld_icon 0
	global modules
	lappend modules [modname]
	global mainCanvasWidth mainCanvasHeight
	
	set modframe $canvas.module[modname]
	frame $modframe -relief raised -borderwidth 3 
	
	bind $modframe <1> "moduleStartDrag $canvas [modname] %X %Y"
	bind $modframe <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $modframe <ButtonRelease-1> "moduleEndDrag $modframe $canvas"
	bind $modframe <3> "popup_menu %X %Y $canvas $minicanvas [modname]"
	
	frame $modframe.ff
	pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
	 
	set p $modframe.ff

	if {[have_ui]} {
	    global ui_font
	    button $p.ui -text "UI" -borderwidth 2 -font $ui_font \
		-anchor center -command "$this initialize_ui"
	    pack $p.ui -side left -ipadx 5 -ipady 2
	}
	global modname_font
	global time_font

	#  Make the mini module icon on a particular canvas
	set miniframe $minicanvas.module[modname]
	frame $miniframe -borderwidth 0
	frame $miniframe.ff
	pack $miniframe.ff -side left -expand yes -fill both -padx 2 -pady 1

	global SCALEX SCALEY
	global basecolor
	$minicanvas create rectangle [expr $modx/$SCALEX] [expr $mody/$SCALEY]\
		[expr $modx/$SCALEX + 4] [expr $mody/$SCALEY + 2] \
		-outline "" -fill $basecolor -tags [modname]

	# Make the title
	label $p.title -text "$name" -font $modname_font -anchor w
	if {$make_progress_graph} {
	    pack $p.title -side top -padx 2 -anchor w
	} else {
	    pack $p.title -side left -padx 2 -anchor w
	}
	bind $p.title <Map> "$this setDone"
	
	# Make the time label
	if {$make_time} {
	    label $p.time -text "00.00" -font $time_font
	    pack $p.time -side left -padx 2
	}

	# Make the progress graph
	if {$make_progress_graph} {
	    frame $p.inset -relief sunken -height 4 \
		-borderwidth 2 -width $graph_width
	    pack $p.inset -side left -fill y -padx 2 -pady 2
	    frame $p.inset.graph -relief raised \
		-width 0 -borderwidth 2 -background green
	}

	# Make the message indicator
	if {!$make_progress_graph} {
	    # No progress graph so pack next to title
	    frame $p.msg -relief sunken -height 15 -borderwidth 1 \
		-width [expr $indicator_width+2]
	    pack $p.msg -side right  -padx 2 -pady 2
	    frame $p.msg.indicator -relief raised -width 0 -height 0 \
		-borderwidth 2 -background blue
	} else {
	    frame $p.msg -relief sunken -height 15 -borderwidth 1 \
		-width [expr $indicator_width+2]
	    pack $p.msg -side left  -padx 2 -pady 2
	    frame $p.msg.indicator -relief raised -width 0 -height 0 \
		-borderwidth 2 -background blue
	}
	bind $p.msg.indicator <Button> "$this displayLog"

	update_msg_state
	update_progress
	update_time

	# Stick it in the canvas
	$canvas create window $modx $mody -window $modframe \
	    -tags [modname] -anchor nw 

	# Set up input/output ports
	$this configureIPorts $canvas
	$this configureOPorts $canvas
	
	# Try to find a position for the icon where it doesn't
	# overlap other icons
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
		    $minicanvas coords [modname] \
			[expr $modx / $SCALEX] [expr $mody / $SCALEY] \
			[expr ($modx+120) / $SCALEX] [expr ($mody+50) /$SCALEY]
		}
	    }
	}
	
	menu $p.menu -tearoff false -disabledforeground white

	# Destroy selected items with a Ctrl-D press
	bind all <Control-d> "moduleDestroySelected $canvas $minicanvas"
	# Clear the canvas
	bind all <Control-l> "ClearCanvas"

	bind all <Control-z> "undo"
	bind all <Control-y> "redo"
        
	# Select the clicked item, and unselect all others
	bind $p <2> "$this toggleSelected $canvas 0"
	bind $p.title <2> "$this toggleSelected $canvas 0"
	if {$make_time} {
	    bind $p.time <2> "$this toggleSelected $canvas 0"
	    bind $p.inset <2> "$this toggleSelected $canvas 0" 
	}
	if {[have_ui]} {
	    bind $p.ui <2> "$this toggleSelected $canvas 0"
	}

	# Select the item in focus, leaving all others selected
	bind $p <Control-Button-2> "$this toggleSelected $canvas 1"
	bind $p.title <Control-Button-2> "$this toggleSelected $canvas 1"
	if {$make_time} {
	    bind $p.time <Control-Button-2> "$this toggleSelected $canvas 1"
	    bind $p.inset <Control-Button-2> "$this toggleSelected $canvas 1"
	}
	if {[have_ui]} {
	    bind $p.ui <Control-Button-2> "$this toggleSelected $canvas 1"
	}
	 
	# Select the item in focus, and unselect all others
	bind $canvas <2> "startBox %X %Y $canvas 0"
	bind $canvas <Control-Button-2> "startBox %X %Y $canvas 1"
	bind $canvas <B2-Motion> "makeBox %X %Y $canvas"
	bind $canvas <ButtonRelease-2> "endBox %X %Y $canvas"

	bind $p <1> "$canvas raise $this"

	bindtags $p [linsert [bindtags $p] 1 $modframe]
	bindtags $p.title [linsert [bindtags $p.title] 1 $modframe]
	if {$make_time} {
	    bindtags $p.time [linsert [bindtags $p.time] 1 $modframe]
	}
	if {$make_progress_graph} {
	    bindtags $p.inset [linsert [bindtags $p.inset] 1 $modframe]
	}
	set made_icon 1
    }
    
    method set_moduleConnected { ModuleConnected } {
	set mconnected $ModuleConnected
    }
    
    method get_moduleConnected {} {
	return $mconnected
    }

    method configureIPorts {canvas} {
	set modframe $canvas.module[modname]
	set i 0
	set temp "a"
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
	    set port $modframe.iport$i
	    bevel $port -width $port_width -height $port_height \
		-borderwidth 3 -edge $e -background $portcolor \
		-pto 2 -pwidth 7 -pborder 2
	    place $port -bordermode outside -x $x -y 0 -anchor nw
	    frame $modframe.iportlight$i -width $port_width -height 4 \
		    -relief raised -background black -borderwidth 0
	    place $modframe.iportlight$i -in $modframe.iport$i \
		    -x 0 -rely 1.0 -anchor nw
	    bind $port <2> "startIPortConnection [modname] $i %x %y"
	    bind $port <B2-Motion> "trackIPortConnection [modname] $i %x %y"
	    bind $port <ButtonRelease-2> "endPortConnection"
	    bind $port <ButtonPress-1> "IPortTrace [modname] $i $temp"
	    bind $port <ButtonRelease-1> "IPortReset $temp"
	    incr i
	} 
	rebuildConnections [netedit getconnected [modname]] 0
    }

    method configureOPorts {canvas} {
	set modframe $canvas.module[modname]
	set temp "a"
	set i 0
	while {[winfo exists $modframe.oport$i]} {
	    destroy $modframe.oport$i
	    destroy $modframe.oportlight$i
	    incr i
	}
	set portinfo [$this-c oportinfo]
	set i 0
	global port_spacing port_width port_height
	foreach t $portinfo {
	    set portcolor [lindex $t 0]
	    set connected [lindex $t 1]
	    set x [expr $i*$port_spacing+6]
	    if {$connected} {
		set e "outbottom"
	    } else {
		set e "bottom"
	    }
	    set port $modframe.oport$i
	    bevel $port -width $port_width -height $port_height \
		    -borderwidth 3 -edge $e -background $portcolor \
		    -pto 2 -pwidth 7 -pborder 2
	    place $port -bordermode ignore -rely 1 -anchor sw -x $x
	    frame $modframe.oportlight$i -width $port_width -height 4 \
		    -relief raised -background black -borderwidth 0
	    place $modframe.oportlight$i -in $port -x 0 -y 0 -anchor sw
	    bind $port <2> "startOPortConnection [modname] $i %x %y"
	    bind $port <B2-Motion> "trackOPortConnection [modname] $i %x %y"
	    bind $port <ButtonRelease-2> "endPortConnection"
	    bind $port <ButtonPress-1> "OPortTrace [modname] $i $temp"
	    bind $port <ButtonRelease-1> "OPortReset $temp"
	    incr i
	}
	rebuildConnections [netedit getconnected [modname]] 0
    }
   
    method setColorAndTitle {canvas color args} {
	set m [modname]
	$canvas.module$m configure -background $color
	$canvas.module$m.ff configure -background $color
	$canvas.module$m.ff.title configure -background $color
	if {[$m have_ui]} {
	    $canvas.module$m.ff.ui configure -background $color
	}
	if {$make_time} {
	    $canvas.module$m.ff.time configure -background $color
	}
	
	if {[llength $args] < 1} {
	    $canvas.module$m.ff.title configure -text $name -justify left
	} else {
	    $canvas.module$m.ff.title configure -text $args -justify left
	}	
    }
       
    method addSelected {canvas color} {
	if {![$this is_selected]} { 
	    global CurrentlySelectedModules
	    lappend CurrentlySelectedModules [modname]
	    setColorAndTitle $canvas $color
	}
    }    

    method removeSelected {canvas color} {
	if {[$this is_selected]} {
	    #Remove me from the Currently Selected Module List
	    global CurrentlySelectedModules
	    set pos [lsearch $CurrentlySelectedModules [modname]]
	    set CurrentlySelectedModules \
		[lreplace $CurrentlySelectedModules $pos $pos]
	    setColorAndTitle $canvas $color
	}
    }
    
    method toggleSelected { canvas option } {
	global CurrentlySelectedModules
	global selected_color
	if { $option == 0 } {
	    unselectAll
	}
	addSelected $canvas $selected_color
    }
    
    method lightOPort {which color} {
	global maincanvas
	set p $maincanvas.module[modname].oportlight$which
	if {[winfo exists $p]} {
	    $p configure -background $color
	}
    }

    method lightIPort {which color} {
	global maincanvas
	set p $maincanvas.module[modname].iportlight$which
	if {[winfo exists $p]} {
	    $p configure -background $color
	}
    }
  
    method update_progress {} {
	if {!$make_progress_graph} return
	set width [expr int($progress*($graph_width-4))]
	if {$width == $old_width} return
	global maincanvas
	set modframe $maincanvas.module[modname]
	if {$width == 0} {
	    place forget $modframe.ff.inset.graph
	} else {
	    $modframe.ff.inset.graph configure -width $width
	    if {$old_width == 0} {
		place $modframe.ff.inset.graph -relheight 1 \
		    -anchor nw
	    }
	}
	set old_width $width
    }
	
    method update_time {} {
	if {!$make_time} return

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
	global maincanvas
	$maincanvas.module[modname].ff.time configure -text $tstr
    }

    method update_state {} {
	if {!$make_progress_graph} return
	if {$state == "JustStarted 1123"} {
	    set progress 0.5
	    set color red
	} elseif {$state == "Executing"} {
	    set progress 0
	    set color red
	} elseif {$state == "NeedData"} {
	    set progress 1
	    set color yellow
	} elseif {$state == "Completed"} {
	    set progress 1
	    set color green
	} else {
	    set width 0
	    set color grey75
	    set progress 0
	}
	global maincanvas
	$maincanvas.module[modname].ff.inset.graph configure -background $color
	update_progress
    }

    method light_module { } {
	global maincanvas
	setColorAndTitle $maincanvas "\#f0e68c" "COMPILING"
    }

    method reset_module_color { } {
	global maincanvas
	setColorAndTitle $maincanvas grey75
    }
	
    method update_msg_state {} { 
	if {$msg_state == "Error"} {
	    set p 1
	    set color red
	} elseif {$msg_state == "Warning"} {
	    set p 1
	    set color yellow
	} elseif {$msg_state == "Remark"} {
	    set p 1
	    set color blue
	}  elseif {$msg_state == "Reset"} {
	    set p 1
	    set color grey75
	} else {
	    set p 0
	    set color grey75
	}
	global maincanvas
	set modframe $maincanvas.module[modname]
	place forget $modframe.ff.msg.indicator
	$modframe.ff.msg.indicator configure -width $indicator_width \
	    -background $color
	place $modframe.ff.msg.indicator -relheight 1 -anchor nw 
    }

    method get_x {} {
	global maincanvas
	return [lindex [$maincanvas coords [modname]] 0]
    }

    method get_y {} {
	global maincanvas
	return [lindex [$maincanvas coords [modname]] 1]
    }

    method get_this {} {
	return $this
    }

     method get_this_c {} {
	return $this-c
    }

    method is_selected {} {
	global CurrentlySelectedModules
	if {[lsearch $CurrentlySelectedModules [modname]] != -1} {
	    return 1
	}
	return 0
    }

    method displayLog {} {
	set w .mLogWnd[modname]
	
	# does the window exist?
	if [winfo exists $w] {
	    raise $w
	    return;
	}
	
	# create the window
	toplevel $w
	append t "Log for " [modname]
	set t "$t -- pid=[$this-c getpid]"
	wm title $w $t
	
	frame $w.log
	text $w.log.txt -relief sunken -bd 2 -yscrollcommand "$w.log.sb set"
	scrollbar $w.log.sb -relief sunken -command "$w.log.txt yview"
	pack $w.log.txt $w.log.sb -side left -padx 5 -pady 5 -fill y

	frame $w.fbuttons 
	button $w.fbuttons.ok -text "OK" -command "$this destroyStreamOutput $w"
	
	pack $w.log $w.fbuttons -side top -padx 5 -pady 5
	pack $w.fbuttons.ok -side right -padx 5 -pady 5 -ipadx 3 -ipady 3

	$msgLogStream registerOutput $w.log.txt
    }
    
    method destroyStreamOutput {w} {
	# TODO: unregister only for streams with the supplied output
	$msgLogStream unregisterOutput
	destroy $w
    }
    
    method updateStream {varName} {
	# triggering stream flush
	set $varName
    }
    
    method module_grow { args } {
	set ports [llength [$this-c iportinfo]]
	incr ports
	# thread problem - if mod_width=1, then
	# module isn't done being created and it
	# adds on too many extra_ports
	if {[set $this-done_bld_icon]} {
	    global maincanvas
	    global port_spacing
	    set curwidth [winfo width $maincanvas.module[modname]]
	    set width [expr 8+$ports*$port_spacing] 
	    if {$curwidth < $width } {
		$maincanvas itemconfigure [modname] -width $width
	    }
	}
    }
    
    method module_shrink {} {
	set ports [llength [$this-c iportinfo]]
	global maincanvas
	global port_spacing
	set curwidth [winfo width $maincanvas.module[modname]]
	set width [expr 8+$ports*$port_spacing]
	if {$width < $initial_width} {
	    set width $initial_width
	}
	if {$width < $curwidth } {
	    $maincanvas itemconfigure [modname] -width $width
	}
    }

    method setDone {} {
	#module actually mapped to the canvas
	if {[set $this-done_bld_icon] == 0 } {
	    global maincanvas
	    set $this-done_bld_icon 1	
	    set initial_width [winfo width $maincanvas.module[modname]]
	    module_grow
	}
    }
}   

proc popup_menu {x y canvas minicanvas modid} {
    global CurrentlySelectedModules
    global menu_id
    set menu_id "$canvas.module$modid.ff.menu"
    regenMenu $modid $menu_id $canvas $minicanvas 
    # popup the menu
    tk_popup $menu_id $x $y    
}

proc regenMenu {modid menu_id canvas minicanvas} {
    # Wipe the menu clean...
    for {set c 0} {$c <= 10 } {incr c } {
	$menu_id delete $c
    }
    set thisc [$modid get_this_c]
    set name [$modid name]
    $menu_id add command -label "$modid" -state disabled
    $menu_id add separator
    $menu_id add command -label "Execute" -command "$thisc needexecute"
    $menu_id add command -label "Help" -command "moduleHelp $modid"
    $menu_id add command -label "Notes" -command "moduleNotes $name $modid"
    if [$modid is_selected] { 
	$menu_id add command -label "Destroy Selected" \
	    -command "moduleDestroySelected $canvas $minicanvas"
    }
    $menu_id add command -label "Destroy" \
	-command "moduleDestroy $canvas $minicanvas $modid"
    $menu_id add command -label "Show Log" -command "$modid displayLog"

}

proc startIPortConnection {imodid iwhich x y} {
    global maincanvas 
    set coords [computeIPortCoords $imodid $iwhich]
    set typename [lindex [lindex [$imodid-c iportinfo] $iwhich] 2]
    set portname [lindex [lindex [$imodid-c iportinfo] $iwhich] 3]
    set fullname $typename:$portname
    frame $maincanvas.frame
    label $maincanvas.frame.label -text $fullname -foreground white -bg #036
    pack $maincanvas.frame $maincanvas.frame.label    
    $maincanvas create window [lindex $coords 0] [lindex $coords 1] \
	-window $maincanvas.frame -anchor sw -tags "tempname" 

    # Find all of the OPorts of the same type and draw a temporary line
    # to them....    
    global new_conn_oports
    set new_conn_oports [netedit findoports $imodid $iwhich]
    foreach i $new_conn_oports {
	set omodid [lindex $i 0]
	set owhich [lindex $i 1]	
	set path [routeConnection $omodid $owhich $imodid $iwhich]
	eval $maincanvas create line $path -width 2 \
	    -tags \"tempconnections iconn$owhich$omodid\"
    }
    global potential_connection
    set potential_connection ""
}

proc startOPortConnection {omodid owhich x y} {
    global maincanvas
    set coords [computeOPortCoords $omodid $owhich]
    set typename [lindex [lindex [$omodid-c oportinfo] $owhich] 2]
    set portname [lindex [lindex [$omodid-c oportinfo] $owhich] 3]
    set fullname $typename:$portname
    frame $maincanvas.frame
    label $maincanvas.frame.label -text $fullname -foreground white -bg #036
    pack $maincanvas.frame $maincanvas.frame.label    
    $maincanvas create window [lindex $coords 0] [lindex $coords 1] \
	-window $maincanvas.frame -anchor nw -tags "tempname" 

    # Find all of the IPorts of the same type and draw a temporary line
    # to them....    
    global new_conn_iports
    set new_conn_iports [netedit findiports $omodid $owhich]
    foreach i $new_conn_iports {
	set imodid [lindex $i 0]
	set iwhich [lindex $i 1]	
	set path [routeConnection $omodid $owhich $imodid $iwhich]
	eval $maincanvas create line $path -width 2 \
	    -tags \"tempconnections oconn$iwhich$imodid\"
    }
    global potential_connection
    set potential_connection ""
}


proc buildConnection {connid portcolor omodid owhich imodid iwhich} {
    global maincanvas minicanvas
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    set temp "a"

    eval $maincanvas create bline $path -width 7 -borderwidth 2 -fill \"$portcolor\" -tags $connid

    global $connid-block
    set $connid-block 0
    $maincanvas bind $connid <ButtonRelease-2> "block_pipe $connid $omodid $owhich $imodid $iwhich $portcolor"
    $maincanvas bind $connid <ButtonPress-3> "destroyConnection $connid $omodid $imodid 1"
    $maincanvas bind $connid <ButtonPress-1> "lightPipe $temp $omodid $owhich $imodid $iwhich"
    $maincanvas bind $connid <ButtonRelease-1> "resetPipe $temp $omodid $imodid"
    $maincanvas bind $connid <Control-Button-1> "raisePipe $connid"

    eval $minicanvas create line [scalePath $path] -width 1 -fill \"$portcolor\" -tags $connid

    $minicanvas lower $connid
}

proc IPortTrace { imodid which temp } {
    set connInfo [netedit getconnected $imodid] 
    foreach t $connInfo {
	set fromName [lindex $t 1]
	set fromPort [lindex $t 2]
	set toName [lindex $t 3] 
	set toPort [lindex $t 4]
	if { [string match $toName $imodid] && \
		[string match $which $toPort] } {
	    # light up the pipe
	    global maincanvas minicanvas
	    set path [routeConnection $fromName $fromPort $toName $toPort]
	    eval $maincanvas create bline $path -width 7 \
		    -borderwidth 2 -fill red -tags $temp
	    eval $minicanvas create bline [scalePath $path] -width 1 \
		    -fill red -tags $temp
	}
    }
}

proc IPortReset { temp } {
    global netedit_canvas netedit_mini_canvas
    $netedit_canvas delete $temp
    $netedit_mini_canvas delete $temp
}

proc OPortTrace { omodid which temp } {
    set connInfo [netedit getconnected $omodid]
    set fromName ""
    set fromPort ""
    foreach t $connInfo {
	set fromName [lindex $t 1]
	set fromPort [lindex $t 2]
	set toName [lindex $t 3] 
	set toPort [lindex $t 4]
	if { [string match $fromName $omodid] && \
		[string match $which $fromPort] } {
	    # light up the pipe
	    global maincanvas minicanvas
	    set path [routeConnection $fromName $fromPort $toName $toPort]
	    eval $maincanvas create bline $path -width 7 \
		    -borderwidth 2 -fill red -tags $temp
	    eval $minicanvas create bline [scalePath $path] -width 1 \
		    -fill red -tags $temp
	}
    }
}

proc OPortReset { temp } {
    global netedit_canvas netedit_mini_canvas
    $netedit_canvas delete $temp
    $netedit_mini_canvas delete $temp
}
  
proc block_pipe { connid omodid owhich imodid iwhich pcolor} {
    global netedit_canvas
    global netedit_mini_canvas
    global $connid-block

    if {[expr [set $connid-block] == 0]} {
	eval $netedit_canvas itemconfigure $connid -width 3 -fill gray
	eval $netedit_mini_canvas itemconfigure $connid -fill gray
	incr $connid-block
	netedit blockconnection $connid
    } else {
	eval $netedit_canvas itemconfigure $connid -width 7 -fill $pcolor
	eval $netedit_mini_canvas itemconfigure $connid -fill $pcolor
	set $connid-block 0
	netedit unblockconnection $connid
    }
    $netedit_mini_canvas raise $connid
}

proc lightPipe { temp omodid owhich imodid iwhich } {
    global maincanvas minicanvas
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    eval $maincanvas create bline $path -width 7 \
	-borderwidth 2 -fill red  -tags $temp
    eval $minicanvas create line [scalePath $path] -width 1 \
	-fill red -tags $temp

    eval $minicanvas itemconfigure $omodid -fill green
    eval $minicanvas itemconfigure $imodid -fill green

    $minicanvas raise $temp
}

proc resetPipe { temp omodid imodid } {
    global netedit_canvas
    $netedit_canvas delete $temp
    global netedit_mini_canvas
    $netedit_mini_canvas delete $temp
    global basecolor
    eval $netedit_mini_canvas itemconfigure $omodid -fill $basecolor
    eval $netedit_mini_canvas itemconfigure $imodid -fill $basecolor
    #destroy $netedit_mini_canvas.frame
}

proc raisePipe { connid } {
    global netedit_canvas netedit_mini_canvas
    $netedit_canvas raise $connid
    $netedit_mini_canvas raise $connid
}


# set the optional args command to anything to record the undo action
proc addConnection {omodid owhich imodid iwhich args } {
    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
    if {"" == $connid} {
	tk_messageBox -type ok -parent . -message \
	    "Invalid connection found while loading network: addConnection $omodid $owhich $imodid $iwhich -- discarding." \
	    -icon warning
	return
    }
    set portcolor [lindex [lindex [$omodid-c oportinfo] $owhich] 0]    
    buildConnection $connid $portcolor $omodid $owhich $imodid $iwhich

    global maincanvas
    $omodid configureOPorts $maincanvas
    $imodid configureIPorts $maincanvas

    #if we got here from undo, record this action as undoable
    if [llength $args] {
	global undoList redoList
	lappend undoList [list "add_connection" $connid]
	# new actions invalidate the redo list
	set redoList ""	
    }

    update idletasks
}


# set the optional args command to anything to record the undo action
proc destroyConnection {connid omodid imodid args} { 
    global maincanvas minicanvas
    $maincanvas delete $connid
    $minicanvas delete $connid
    netedit deleteconnection $connid $omodid 
    $omodid configureOPorts $maincanvas
    $imodid configureIPorts $maincanvas

    #if we got here from undo, record this action as undoable
    if [llength $args] {
	global undoList redoList
	lappend undoList [list "sub_connection" $connid]
	# new actions invalidate the redo list
	set redoList ""
    }
}

proc scalePath { path } {
    set minipath ""
    global SCALEX SCALEY
    set doingX 1
    foreach point $path {
	if $doingX {
	    lappend minipath [expr round($point/$SCALEX)] 
	} else {
	    lappend minipath [expr round($point/$SCALEY)] 
	}
	set doingX [expr !$doingX]
    }
    return $minipath
}
    
	
proc rebuildConnection { connid omodid owhich imodid iwhich} {    
    global maincanvas minicanvas
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    eval $maincanvas coords $connid $path
    eval $minicanvas coords $connid [scalePath $path]
}

proc rebuildConnections {list color} {
    global maincanvas
    foreach i $list {
	set id [lindex $i 0]
	$maincanvas raise $id	
	if {$color} {
	    $minicanvas itemconfigure $id -fill [$maincanvas itemcget $id -fill]
	}
	eval rebuildConnection $i
    }
}

proc trackIPortConnection {imodid which x y} {
    global new_conn_oports
    if ![llength $new_conn_oports] return
    # Get coords in canvas
    global maincanvas
    set ox1 [winfo x $maincanvas.module$imodid.iport$which]
    set ox2 [lindex [$maincanvas coords $imodid] 0]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $maincanvas.module$imodid.iport$which]
    set oy2 [lindex [$maincanvas coords $imodid] 1]
    set y [expr $y+$oy1+$oy2]
    set c [computeIPortCoords $imodid $which]
    set ix [lindex $c 0]
    set iy [lindex $c 1]
    set mindist [computeDist $x $y $ix $iy]
    set minport ""
    foreach i $new_conn_oports {
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
    $maincanvas itemconfigure tempconnections -fill black

    global potential_connection
    if {$minport != ""} {
	set omodid [lindex $minport 0]
	set owhich [lindex $minport 1]
	$maincanvas itemconfigure iconn$owhich$omodid -fill red
	set potential_connection [list $omodid $owhich $imodid $which]
    } else {
	set potential_connection ""
    }
}

proc trackOPortConnection {omodid which x y} {
    global new_conn_iports
    if ![llength $new_conn_iports] return
    # Get coords in canvas
    global maincanvas
    set ox1 [winfo x $maincanvas.module$omodid.oport$which]
    set ox2 [lindex [$maincanvas coords $omodid] 0]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $maincanvas.module$omodid.oport$which]
    set oy2 [lindex [$maincanvas coords $omodid] 1]
    set y [expr $y+$oy1+$oy2]
    set c [computeOPortCoords $omodid $which]
    set ix [lindex $c 0]
    set iy [lindex $c 1]
    set mindist [computeDist $x $y $ix $iy]
    set minport ""
    foreach i $new_conn_iports {
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
    $maincanvas itemconfigure tempconnections -fill black

    global potential_connection
    if {$minport != ""} {
	set imodid [lindex $minport 0]
	set iwhich [lindex $minport 1]
	$maincanvas raise oconn$iwhich$imodid
	$maincanvas itemconfigure oconn$iwhich$imodid -fill red
	set potential_connection [list $omodid $which $imodid $iwhich]
    } else {
	set potential_connection ""
    }
}

proc endPortConnection {} {
    global maincanvas
    $maincanvas delete tempconnections
    $maincanvas delete tempname
    destroy $maincanvas.frame
    global potential_connection
    if { $potential_connection != "" } {
	eval addConnection $potential_connection 1
    }
}


proc parseConnectionID {conn} {
    set index1 [string first "_p" $conn ]
    set omodid [string range $conn 0 [expr $index1-1]]
    set index2 [string first "_to_" $conn ]
    set owhich [string range $conn [expr $index1+2] [expr $index2-1]]
    set index4 [string last "_p" $conn]
    set index3 [string last "_" $conn $index4]
    set imodid [string range $conn [expr $index2+4] [expr $index3-1]]
    set iwhich [string range $conn [expr $index4+2] end]
    return "$omodid $owhich $imodid $iwhich"
}




proc undo {} {
    global undoList redoList
    if ![llength $undoList] {
	return
    }
    # Get the last action performed
    set undo_item [lindex $undoList end]
    # Remove it from the list
    set undoList [lreplace $undoList end end]
    # Add it to the redo list
    lappend redoList $undo_item

    set action [lindex $undo_item 0]
    set connid [lindex $undo_item 1]
    set conn_info [parseConnectionID $connid]
    if { $action == "add_connection" } {
	eval destroyConnection $connid [lindex $conn_info 0] [lindex $conn_info 2]
    }
    if { $action == "sub_connection" } {
	eval addConnection $conn_info
    }
}


proc redo {} {
    global undoList redoList
    if ![llength $redoList] {
	return
    }
    # Get the last action undone
    set redo_item [lindex $redoList end]
    # Remove it from the list
    set redoList [lreplace $redoList end end]
    # Add it to the undo list
    lappend undoList $redo_item

    set action [lindex $redo_item 0]
    set connid [lindex $redo_item 1]
    set conn_info [parseConnectionID $connid]
    if { $action == "add_connection" } {
	eval addConnection $conn_info
    }
    if { $action == "sub_connection" } {
	eval destroyConnection $connid [lindex $conn_info 0] [lindex $conn_info 2]
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
    if {$ox == $ix && $oy < $iy} {
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

proc computePortCoords {modid which isoport} {
    global maincanvas
    global port_spacing
    global port_width
    set at [$maincanvas coords $modid]
    set x [expr $which*$port_spacing+6+$port_width/2+[lindex $at 0]]
    if {$isoport} {
	set y [expr [winfo height $maincanvas.module$modid]+[lindex $at 1]]
    } else {
	set y [lindex $at 1]
    }
    return [list $x $y]
}

proc computeIPortCoords {modid which} {
    return [computePortCoords $modid $which 0]
}

proc computeOPortCoords {modid which} {
    return [computePortCoords $modid $which 1]
}


proc computeDist {x1 y1 x2 y2} {
    set dx [expr $x2-$x1]
    set dy [expr $y2-$y1]
    return [expr sqrt($dx*$dx+$dy*$dy)]
}

proc moduleStartDrag {maincanvas modid x y} {
    #raise the module icon
    set wname "$maincanvas.module$modid"
    raise $wname

    #set module movement coordinates
    global lastX lastY
    set lastX $x
    set lastY $y
       
    #if clicked module isnt selected, unselect all and select this
    if { ! [$modid is_selected] } {
	$modid toggleSelected $maincanvas 0
    }

    #build connection lists for all selected modules to draw pipes when moving
    global CurrentlySelectedModules
    foreach csm $CurrentlySelectedModules {
	$csm set_moduleConnected [netedit getconnected $csm]
    }
    
    #create a gray bounding box around moving modules
    global sel_module_box
    set sel_module_box [$maincanvas create rectangle [compute_bbox $maincanvas]]
    $maincanvas itemconfigure $sel_module_box -outline darkgray
}

proc moduleDrag {maincanvas minicanvas modid x y} {
    global grouplastX
    global grouplastY
    global lastX
    global lastY

    set bbox [compute_bbox $maincanvas]
    # When the user tries to drag a group of modules off the canvas,
    # Offset the lastX and or lastY variable, so that they can only drag
    #groups to the border of the canvas
    set min_possibleX [expr [lindex $bbox 0] + ($x-$lastX)]
    set min_possibleY [expr [lindex $bbox 1] + ($y-$lastY)]
    
    if {$min_possibleX <= 0} {
	set lastX [expr $lastX+$min_possibleX]
    }
    if {$min_possibleY <= 0} {
	set lastY [expr $lastY+$min_possibleY]
    }
    
    set max_possibleX [expr [lindex $bbox 2] + ($x-$lastX)]
    set max_possibleY [expr [lindex $bbox 3] + ($y-$lastY)]
    
    if {$max_possibleX >= 4500} {
	set diff [expr $max_possibleX-4500]
	set lastX $lastX-$diff
    }
    if {$max_possibleY >= 4500} {
	set diff [expr $max_possibleY-4500]
	set lastY $lastY-$diff
    }
    
    # Move each module individually and redraw all connections
    global CurrentlySelectedModules
    foreach csm $CurrentlySelectedModules {
	do_moduleDrag $maincanvas $minicanvas $csm $x $y
	rebuildConnections [$csm get_moduleConnected] 0
    }	
    
    set lastX $grouplastX
    set lastY $grouplastY
    
    global sel_module_box
    $maincanvas coords $sel_module_box [compute_bbox $maincanvas]
}    

proc do_moduleDrag {maincanvas minicanvas modid x y} {
    global xminwarped
    global xmaxwarped
    global yminwarped
    global ymaxwarped
    global lastX lastY
    global SCALEX SCALEY
    global CurrentlySelectedModules
    global sel_module_box

    set templastX $lastX
    set templastY $lastY
        
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
    set currybdr [expr $canheight+ ([lindex [$maincanvas yview] 0]*$mainCanvasHeight)]

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
	    #if not on edge of canvas, move viewable area right	 
	    set xbegView [lindex [$maincanvas xview] 0]
	    set xdiff [expr ($mmodxpos+($x-$lastX))-$currxbdr]
	    set mvx [expr (($xdiff/$mainCanvasWidth)+$xbegView)]
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
    #####################################################################
        
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
    
    global grouplastX
    global grouplastY
    
    set grouplastX $lastX
    set grouplastY $lastY

    set lastX $templastX
    set lastY $templastY
}

proc moduleEndDrag {mframe maincanvas} {
    global sel_module_box
    $maincanvas delete $sel_module_box
    computeModulesBbox
}

proc moduleHelp {modid} {
    set w .mHelpWindow[$modid name]
	
    # does the window exist?
    if [winfo exists $w] {
	raise $w
	return;
    }
	
    # create the window
    toplevel $w
    append t "Help for " [$modid name]
    wm title $w $t
	
    frame $w.help
    text $w.help.txt -relief sunken -wrap word -bd 2 \
	-yscrollcommand "$w.help.sb set"
    scrollbar $w.help.sb -relief sunken -command "$w.help.txt yview"
    pack $w.help.txt $w.help.sb -side left -padx 5 -pady 5 -fill y

    frame $w.fbuttons 
    button $w.fbuttons.ok -text "Close" -command "destroy $w"
    
    pack $w.help $w.fbuttons -side top -padx 5 -pady 5
    pack $w.fbuttons.ok -side right -padx 5 -pady 5 -ipadx 3 -ipady 3

    $w.help.txt insert end [$modid-c help]
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
    set $mclass-notes [$w.tnotes get 1.0 end]
    destroy $w
}

proc moduleDestroy {maincanvas minicanvas modid} {
    # Remove me from the modules list
    global modules
    set pos [lsearch $modules $modid]
    set modules [lreplace $modules $pos $pos]

    #Remove me from the Currently Selected Module List
    global CurrentlySelectedModules
    set pos [lsearch $CurrentlySelectedModules [$modid modname]]
    set CurrentlySelectedModules [lreplace $CurrentlySelectedModules $pos $pos]
    
    set modList [netedit getconnected $modid]  
    # go through list from backwards to work for
    # dynamic modules also
    set size [expr [llength $modList]-1]
    for {set j $size} {$j >= 0} {incr j -1} {
	set i [lindex $modList $j]
	destroyConnection [lindex $i 0] [lindex $i 1] [lindex $i 3]
    }

    # Hack, work around Viewer deletion bug by waiting a moment to destroy
    # the module after it has been disconnected.
    if {[string first Viewer $modid] != -1} {
	after 100 "moduleDestroyAux $maincanvas $minicanvas $modid"
    } else {
	moduleDestroyAux $maincanvas $minicanvas $modid
    }
}

proc moduleDestroyAux {maincanvas minicanvas modid} {
    $maincanvas delete $modid
    destroy ${maincanvas}.module$modid
    $minicanvas delete $modid
    destroy $minicanvas.module$modid
    netedit deletemodule $modid
    $modid delete
    
    if {[winfo exists .ui$modid]} {
	destroy .ui$modid
    }
    computeModulesBbox
}

proc moduleDestroySelected {maincanvas minicanvas} {
    global CurrentlySelectedModules 
    foreach mnum $CurrentlySelectedModules {
	moduleDestroy $maincanvas $minicanvas $mnum
    }
}

global startx starty rect
global InitiallySelectedModules

proc startBox {X Y maincanvas keepselected} {
    global CurrentlySelectedModules
    global InitiallySelectedModules
    global mainCanvasWidth mainCanvasHeight
    global startx starty rect

    if {!$keepselected} {
	unselectAll
	set InitiallySelectedModules ""
    } else {
	set InitiallySelectedModules $CurrentlySelectedModules
    }
    
    set sx [expr $X-[winfo rootx $maincanvas]]
    set sy [expr $Y-[winfo rooty $maincanvas]]
    
    set startx [expr $sx + \
	int([expr (([lindex [$maincanvas xview] 0]*$mainCanvasWidth))])]
    set starty [expr $sy + \
        int([expr (([lindex [$maincanvas yview] 0]*$mainCanvasHeight))])]
  
    #Begin the bounding box
    set rect [$maincanvas create rectangle $startx $starty $startx $starty] 
    
    global rx ry
    set rx [winfo rootx $maincanvas]
    set ry [winfo rooty $maincanvas]    
}

proc makeBox {X Y maincanvas} {
    global CurrentlySelectedModules
    global InitiallySelectedModules
    global mainCanvasWidth mainCanvasHeight
    global selected_color unselected_color
    global startx starty rx ry
    global rect
    
    #Canvas Relative current X and Y positions
    set currx [expr [expr ($X-$rx)] + \
       int([expr (([lindex [$maincanvas xview] 0]*$mainCanvasWidth))])]
    set curry [expr [expr ($Y-$ry)] + \
       int([expr (([lindex [$maincanvas yview] 0]*$mainCanvasHeight))])]

    #redraw box
    $maincanvas coords $rect $startx $starty $currx $curry    

    # select all modules which overlap the current bounding box
    set overlappingModules ""
    set overlap [$maincanvas find overlapping $startx $starty $currx $curry]
    foreach i $overlap {
	set s "[$maincanvas itemcget $i -tags]"
	if {[$maincanvas type $s] == "window"} {
	    lappend overlappingModules $s
	    if {[lsearch $CurrentlySelectedModules $s] == -1} {
		$s addSelected $maincanvas $selected_color
	    }
	}
    }

    # remove those not initally selected or overlapped by box
    foreach mod $CurrentlySelectedModules {
	if {[lsearch $overlappingModules $mod] == -1 && \
		[lsearch $InitiallySelectedModules $mod] == -1} {
	    $mod removeSelected $maincanvas $unselected_color
	}
    }
}

proc endBox {X Y maincanvas} {
    global rect
    $maincanvas delete $rect
}

proc unselectAll {} {
    global CurrentlySelectedModules maincanvas unselected_color
    foreach i $CurrentlySelectedModules {
	$i removeSelected $maincanvas $unselected_color
    }
}

proc compute_bbox {maincanvas} {
    #Compute and return the coordinated of a bounding box containing all
    #CurrentlySelectedModules
    global CurrentlySelectedModules
    set maxx 0
    set maxy 0
    set minx 4500
    set miny 4500

    foreach csm  $CurrentlySelectedModules {
	set curr_coords [$maincanvas coords $csm]

	#Find $maxx and $maxy
	if { [lindex [$maincanvas bbox $csm] 2] > $maxx} {
	    set maxx [lindex [$maincanvas bbox $csm] 2]
	}
	if { [lindex [$maincanvas bbox $csm] 3] > $maxy} {
	    set maxy [lindex [$maincanvas bbox $csm] 3]
	}

	#Find $minx and $miny
	if { [lindex $curr_coords 0] <= $minx} {
	    set minx [lindex $curr_coords 0]
	}
	if { [lindex $curr_coords 1] <= $miny} {
	    set miny [lindex $curr_coords 1]
	}
    }
    
    return "$minx $miny $maxx $maxy"
}
