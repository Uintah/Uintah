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

global Color
set Color(Selected) LightSkyBlue2
set Color(Disabled) black
set Color(Compiling) "\#f0e68c"
set Color(Trace) red

global CurrentlySelectedModules
set CurrentlySelectedModules ""

global startX
set startX 0

global startY
set startY 0

global modules
set modules ""

global undoList
set undoList ""

global redoList
set redoList ""

global HelpText
set HelpText(Module) "L - Select\nR - Menu"
set HelpText(Connection) "L - Highlight\nCTRL-M - Delete\nR - Menu"
set HelpText(Notes) "L - Edit\nM - Hide"

itcl_class Module {
   
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
			
    constructor {config} {
	set msgLogStream "[TclStream msgLogStream#auto]"
	# these live in parallel temporarily
	global $this-notes Notes
	if ![info exists $this-notes] { set $this-notes "" }

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
    # flag set when the module is compiling
    protected compiling_p 0
    # flag set when the module has all incoming ports blocked
    public state "NeedData" {$this update_state}
    public msg_state "Reset" {$this update_msg_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}

    method compiling_p {} { return $compiling_p }
    method set_compiling_p { val } { 
	set compiling_p $val	
	setColorAndTitle
        if {[winfo exists .standalone]} {
	    app indicate_dynamic_compile [modname] \
		[expr $val?"start":"stop"]
	}

    }

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
	return [llength [$this info method ui]]
    }

    #  Make the modules icon on a particular canvas
    method make_icon {canvas minicanvas modx mody} {
	global $this-done_bld_icon
	set $this-done_bld_icon 0
	global modules
	lappend modules [modname]
	global mainCanvasWidth mainCanvasHeight
	
	global Disabled
	set Disabled([modname]) 0	
	
	set modframe $canvas.module[modname]
	frame $modframe -relief raised -borderwidth 3 
	
	bind $modframe <1> "moduleStartDrag $canvas [modname] %X %Y 0"
	bind $modframe <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $modframe <ButtonRelease-1> "moduleEndDrag $modframe $canvas %X %Y"
	bind $modframe <Control-Button-1> "moduleStartDrag $canvas [modname] %X %Y 1"
	bind $modframe <3> "moduleMenu %X %Y $canvas $minicanvas [modname]"
	
	frame $modframe.ff
	set p $modframe.ff
	pack $p -side top -expand yes -fill both -padx 5 -pady 6

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

	if {[have_ui]} {
	    global ui_font
	    button $p.ui -text "UI" -borderwidth 2 -font $ui_font \
		-anchor center -command "$this initialize_ui"
	    pack $p.ui -side left -ipadx 5 -ipady 2
	}

	# Make the title
	global modname_font
	label $p.title -text "$name" -font $modname_font -anchor w
	pack $p.title -side [expr $make_progress_graph?"top":"left"] \
	    -padx 2 -anchor w
	bind $p.title <Map> "$this setDone"


	# Make the time label
	if {$make_time} {
	    global time_font
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
	frame $p.msg -relief sunken -height 15 -borderwidth 1 \
	    -width [expr $indicator_width+2]
	pack $p.msg -side [expr $make_progress_graph?"left":"right"] \
	    -padx 2 -pady 2
	frame $p.msg.indicator -relief raised -width 0 -height 0 \
	    -borderwidth 2 -background blue
	bind $p.msg.indicator <Button> "$this displayLog"

	update_msg_state
	update_progress
	update_time

	# Stick it in the canvas
	$canvas create window $modx $mody -window $modframe \
	    -tags [modname] -anchor nw 

	# Set up input/output ports
	$this configurePorts $canvas
	
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

	bindtags $p [linsert [bindtags $p] 1 $modframe]
	bindtags $p.title [linsert [bindtags $p.title] 1 $modframe]
	if {$make_time} {
	    bindtags $p.time [linsert [bindtags $p.time] 1 $modframe]
	}
	if {$make_progress_graph} {
	    bindtags $p.inset [linsert [bindtags $p.inset] 1 $modframe]
	}
	update idletasks
    }
    
    method configurePorts {canvas args} {
	if ![llength $args] { set args "i o" }
	foreach porttype $args {

	    global port_spacing port_width port_height
	    set isoport [string equal $porttype o]

	    set modframe $canvas.module[modname]
	    set i 0
	    while {[winfo exists $modframe.port$porttype$i]} {
		destroy $modframe.port$porttype$i
		destroy $modframe.portlight$porttype$i
		incr i
	    }
	    set portinfo [$this-c $porttype.portinfo]
	    set i 0
	    foreach t $portinfo {
		set portcolor [lindex $t 0]
		set portname [join [lrange $t 2 3] ":"]
		set x [expr $i*$port_spacing+6]
		set e [expr $isoport?"bottom":"top"]
		set e "[expr [lindex $t 1]?"out":""]$e"
		set port $modframe.port$porttype$i
		set portlight $modframe.portlight$porttype$i
		bevel $port -width $port_width -height $port_height \
		    -borderwidth 3 -edge $e -background $portcolor \
		    -pto 2 -pwidth 7 -pborder 2
		frame $portlight -width $port_width -height 4 \
		    -relief raised -background black -borderwidth 0
		if $isoport {
		    place $port -bordermode ignore -rely 1 -anchor sw -x $x
		    place $portlight -in $port -x 0 -y 0 -anchor sw
		} else {
		    place $port -bordermode outside -x $x -y 0 -anchor nw
		    place $portlight -in $port -x 0 -rely 1.0 -anchor nw
		}

	       
		bind $port <2> "startPortConnection [modname] $i $porttype \"$portname\" %x %y"
		bind $port <B2-Motion> "trackPortConnection [modname] $i $porttype %x %y"
		bind $port <ButtonRelease-2> "endPortConnection"
		bind $port <ButtonPress-1> "TracePort [modname] $isoport $i"
		bind $port <ButtonRelease-1> "deleteTrace"

		bind $portlight <2> "startPortConnection [modname] $i $porttype \"$portname\" %x %y"
		bind $portlight <B2-Motion> "trackPortConnection [modname] $i $porttype %x %y"
		bind $portlight <ButtonRelease-2> "endPortConnection"
		bind $portlight <ButtonPress-1> "TracePort [modname] $isoport $i"
		bind $portlight <ButtonRelease-1> "deleteTrace"

		global HelpText
		Tooltip $port $HelpText(Connection)
		Tooltip $portlight $HelpText(Connection)
		incr i
	    } 
	}
	rebuildConnections [netedit getconnected [modname]]
    }
   
    method setColorAndTitle {args} {
	global maincanvas Color Disabled basecolor
	set m $maincanvas.module[modname]
	$m configure -relief raised
	set color $basecolor
	if { [$this is_selected] } { set color $Color(Selected) }
	if { $Disabled([modname]) } { 
	    set color [blend $color $Color(Disabled)]
	    $m configure -relief sunken
	}

	if { $compiling_p } {
	    set color $Color(Compiling)
	    set $args "COMPILING"
	}
	$m configure -background $color
	$m.ff configure -background $color
	$m.ff.title configure -background $color
	if {[$this have_ui]} { $m.ff.ui configure -background $color }
	if {$make_time} { $m.ff.time configure -background $color }
	if {![llength $args]} { set args $name }
	$m.ff.title configure -text "$args" -justify left
	update idletasks
    }
       
    method addSelected {canvas} {
	if {![$this is_selected]} { 
	    global CurrentlySelectedModules
	    lappend CurrentlySelectedModules [modname]
	    setColorAndTitle
	}
    }    

    method removeSelected {canvas} {
	if {[$this is_selected]} {
	    #Remove me from the Currently Selected Module List
	    global CurrentlySelectedModules
	    set pos [lsearch $CurrentlySelectedModules [modname]]
	    set CurrentlySelectedModules \
		[lreplace $CurrentlySelectedModules $pos $pos]
	    setColorAndTitle
	}
    }
    
    method toggleSelected { canvas option } {
	global CurrentlySelectedModules
	if { $option == 0 } { unselectAll }
	if [is_selected] { removeSelected $canvas
	} else { addSelected $canvas }
    }
    
    method lightOPort {which color} {
	global maincanvas
	set p $maincanvas.module[modname].portlighto$which
	if {[winfo exists $p]} {
	    $p configure -background $color
	}
    }

    method lightIPort {which color} {
	global maincanvas
	set p $maincanvas.module[modname].portlighti$which
	if {[winfo exists $p]} {
	    $p configure -background $color
	}
    }
  
    method update_progress {} {
	set width [expr int($progress*($graph_width-4))]
	if {!$make_progress_graph || $width == $old_width } return
	global maincanvas
	set graph $maincanvas.module[modname].ff.inset.graph
	if {$width == 0} { 
	    place forget $graph
	} else {
	    $graph configure -width $width
	    if {$old_width == 0} { place $graph -relheight 1 -anchor nw }
	    if {[winfo exists .standalone]} {
		app update_progress [modname] $state
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

	if {[winfo exists .standalone]} {
	    app indicate_error [modname] $msg_state
	}
	
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
	if {[lsearch $CurrentlySelectedModules [modname]] != -1} { return 1 }
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

proc moduleMenu {x y canvas minicanvas modid} {
    set menu_id "$canvas.module$modid.ff.menu"
    regenModuleMenu $modid $menu_id $canvas $minicanvas 
    tk_popup $menu_id $x $y    
}

proc regenModuleMenu {modid menu_id canvas minicanvas} {
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
    $menu_id add command -label "Notes" -command "notesWindow $modid notesDoneModule"
    if [$modid is_selected] { 
	$menu_id add command -label "Destroy Selected" \
	    -command "moduleDestroySelected $canvas $minicanvas"
    }
    $menu_id add command -label "Destroy" \
	-command "moduleDestroy $canvas $minicanvas $modid"
    $menu_id add command -label "Show Log" -command "$modid displayLog"
    global Disabled
    if $Disabled($modid) {
	$menu_id add command -label "Enable" -command "disableModule $modid 0"
    } else {
	$menu_id add command -label "Disable" -command "disableModule $modid 1"
    }
}

# args == { connid omodid owhich imodid iwhich }
proc connectionMenu {x y args} {
    if { [llength $args] != 5 } { return }
    global maincanvas
    set menu_id "$maincanvas.menu[lindex args 0]"
    eval regenConnectionMenu $menu_id $args
    tk_popup $menu_id $x $y    
}

proc regenConnectionMenu { menu_id args } {
    # create menu if it doesnt exist
    if ![winfo exists $menu_id] {
	menu $menu_id -tearoff 0 -disabledforeground white
    }
    # Wipe the menu clean...
    for {set c 0} {$c <= 10 } {incr c } {
	$menu_id delete $c
    }
    $menu_id add command -label "Connection" -state disabled
    $menu_id add separator
    $menu_id add command -label "Delete" -command \
	"destroyConnection [lindex $args 0] [lindex $args 1] [lindex $args 3]"
    global Disabled
    $menu_id add command -command "eval block_pipe $args" \
	-label [expr $Disabled([lindex $args 0])?"Enable":"Disable"]
    set id [lindex $args 0]
    $menu_id add command -label "Notes" -command \
	"notesWindow $id notesDoneConnection"
}


proc notesDoneConnection { id } {
    drawNotes $id
}

proc notesDoneModule { id } {
    global Notes $id-notes
    set $id-notes $Notes($id)
    drawNotes $id
}


proc notesWindow { id done } {
    global Notes Color NotesPos
    if { [winfo exists .notes] } { destroy .notes }
    toplevel .notes
    text .notes.input -relief sunken -bd 2 -height 20
    frame .notes.b
    button .notes.b.done -text "Done" -command "okNotesWindow $id \"$done\""
    button .notes.b.clear -text "Clear" -command ".notes.input delete 1.0 end"
    button .notes.b.cancel -text "Cancel" -command "destroy .notes"

    set rgb [expr [info exists Color($id)]?"$Color($id)":"white"]
    button .notes.b.reset -fg black -text "Reset Color" -command \
	"set Color(Notes-$id) $rgb; .notes.b.color configure -bg $rgb"

    set rgb [expr [info exists Color(Notes-$id)]?"$Color(Notes-$id)":"$rgb"]
    button .notes.b.color -fg black -bg $rgb -text "Text Color" -command \
	"colorNotes $id"

    frame .notes.d -relief groove -borderwidth 2
    if {![info exists NotesPos($id)] } { set NotesPos($id) def }
    make_labeled_radio .notes.d.pos "Display:" "" left NotesPos($id) \
	{
	    { "Default" def } \
		{ "None" none } \
		{ "Tooltip" tooltip } \
		{ "Top" n } \
		{ "Left" w } \
		{ "Right" e } \
		{ "Bottom" s } \
	    }

    pack .notes.input -fill x -side top -padx 5 -pady 3
    pack .notes.d -fill x -side top -padx 5 -pady 0
    pack .notes.d.pos
    pack .notes.b -fill y -side bottom -pady 3
    pack .notes.b.done .notes.b.clear .notes.b.cancel .notes.b.reset \
	.notes.b.color -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
	    
    if [info exists Notes($id)] {.notes.input insert 1.0 $Notes($id)}
}

proc colorNotes { id } {
    global Color    
    .notes.b.color configure -bg [set Color(Notes-$id) \
       [tk_chooseColor -initialcolor [.notes.b.color cget -bg]]]
}

    

proc okNotesWindow { id done } {
    global Notes
    set Notes($id) [.notes.input get 1.0 "end - 1 chars"]
    destroy .notes
    eval $done $id
    update idletasks
}


proc disableModule { module state } {
    global Disabled CurrentlySelectedModules
    set mods [expr [$module is_selected]?"$CurrentlySelectedModules":"$module"]
    foreach modid $mods {
	# iterate through every conneciton into and out of the module
	foreach connectionInfo [netedit getconnected $modid] {
	    # if connection is already equal to state, then dont do anything
	    if { $Disabled([lindex $connectionInfo 0]) == $state } { continue }
	    # otherwise, enable or disable the connection
	    eval block_pipe $connectionInfo
	}
    }
}

proc checkForDisabledModules { args } {
    global Disabled
    # iterate through unique modules to set disabled flag
    foreach modid [lsort -unique [eval list $args]] {
	# assume module is disabled
	set Disabled($modid) 1
	foreach connectionInfo [netedit getconnected $modid] {
	    # if connection is enabled, then enable module
	    if { !$Disabled([lindex $connectionInfo 0]) } {
		set Disabled($modid) 0
		# module is enabled, continue onto next module
		break;
	    }
	}
	$modid setColorAndTitle
    }
}

proc buildConnection {connid portcolor omodid owhich imodid iwhich} {
    global maincanvas minicanvas Color Disabled HelpText
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    eval $maincanvas create bline $path -width 7 -borderwidth 2 \
	-fill \"$portcolor\" -tags $connid
    eval $minicanvas create line [scalePath $path] -width 1 \
	-fill \"$portcolor\" -tags $connid
    $minicanvas lower $connid

    set Disabled($connid) 0
    set Color($connid) $portcolor

    $maincanvas bind $connid <ButtonPress-1> \
	"canvasRaise $connid; TraceConnection $omodid $owhich $imodid $iwhich"
    $maincanvas bind $connid <Control-Button-2> \
	"destroyConnection $connid $omodid $imodid 1"
    $maincanvas bind $connid <ButtonPress-3> \
	"connectionMenu %X %Y $connid $omodid $owhich $imodid $iwhich"
    $maincanvas bind $connid <ButtonRelease> \
	"+deleteTrace $omodid $owhich $imodid $iwhich"
   
    canvasTooltip $connid $HelpText(Connection)
}

# Deletes red connections on canvas and turns port lights black
# args = { omodid owhich imodid iwhich }
proc deleteTrace { args } {
    canvasDelete tempConnection
    if { [llength $args] != 4 } { return }
    [lindex $args 0] lightOPort [lindex $args 1] black
    [lindex $args 2] lightIPort [lindex $args 3] black
}


global TracedPorts
set TracedPorts ""

proc TracePort { modid traceoport { port "all" } } {
    global TracedPorts
    if {$port == "all" } { set TracedPorts "" }
    set portidx [expr $traceoport?1:3]
    foreach conn [netedit getconnected $modid] {
	# if 1. Port not already traced and
	#    2. Connection originates at this module and
	#    3. Either we want to trace all module ports or
	#       $conn is the port we want to trace
	if { [lsearch $TracedPorts [lindex $conn 0]] == -1 && \
	      $modid == [lindex $conn $portidx] && \
	      ($port == "all" || [lindex $conn [expr $portidx+1]] == $port)} {
	    lappend TracedPorts [lindex $conn 0]
	    eval lightPipe tempConnection [lrange $conn 1 4]
	    TracePort [lindex $conn [expr $traceoport?3:1]] $traceoport
	}
    }
}

proc TraceConnection { args } {
    if { [llength $args] != 4 } return
    eval lightPipe tempConnection $args
    global TracedPorts Color
    set TracedPorts ""
    [lindex $args 0] lightOPort [lindex $args 1] $Color(Trace)
    [lindex $args 2] lightIPort [lindex $args 3] $Color(Trace)
    TracePort [lindex $args 0] 0
    TracePort [lindex $args 2] 1
}

proc canvasExists { arg } {
    global maincanvas
    return [expr [llength [$maincanvas find withtag $arg]]?1:0]
}

proc canvasDelete { args } {
    global maincanvas minicanvas
    eval $maincanvas delete $args
    eval $minicanvas delete $args
}

proc canvasRaise { args } {
    global maincanvas minicanvas
    foreach arg $args {
	$maincanvas raise $arg
	$minicanvas raise $arg
    }
}

proc block_pipe { connid omodid owhich imodid iwhich } {
    global maincanvas minicanvas Disabled Color
    if {!$Disabled($connid)} {
        $maincanvas itemconfigure $connid -width 3 -fill gray
	$minicanvas itemconfigure $connid -fill gray
	set Color(Notes-$connid) gray
	drawNotes $connid
	set Disabled($connid) 1
	netedit blockconnection $connid
    } else {
	$maincanvas itemconfigure $connid -width 7 -fill $Color($connid)
	$minicanvas itemconfigure $connid -fill $Color($connid)
	set Color(Notes-$connid) $Color($connid)
	drawNotes $connid
	set Disabled($connid) 0
	netedit unblockconnection $connid
    }
    checkForDisabledModules $imodid $omodid
    canvasRaise $connid
}

proc lightPipe { temp omodid owhich imodid iwhich } {
    global maincanvas minicanvas Color
    set path [routeConnection $omodid $owhich $imodid $iwhich]
    eval $maincanvas create bline $path -width 7 \
	-borderwidth 2 -fill $Color(Trace)  -tags $temp
    eval $minicanvas create line [scalePath $path] -width 1 \
	-fill $Color(Trace) -tags $temp
    $minicanvas itemconfigure $omodid -fill green
    $minicanvas itemconfigure $imodid -fill green
    canvasRaise $temp
}

# set the optional args command to anything to record the undo action
proc addConnection {omodid owhich imodid iwhich args } {
    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
    if {"" == $connid} {
	tk_messageBox -type ok -parent . -icon wanring -message \
	    "Invalid connection found while loading network: addConnection $omodid $owhich $imodid $iwhich -- discarding." 
	return
    }
    set portcolor [lindex [lindex [$omodid-c oportinfo] $owhich] 0]    
    buildConnection $connid $portcolor $omodid $owhich $imodid $iwhich
    if ![info exists Notes($connid)] { set Notes($connid) "" }
    global maincanvas
    $omodid configurePorts $maincanvas o
    $imodid configurePorts $maincanvas i

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
    global maincanvas
    deleteTrace
    canvasDelete $connid
    netedit deleteconnection $connid $omodid 
    $omodid configurePorts $maincanvas o
    $imodid configurePorts $maincanvas i
    if { [canvasExists $connid-notes] } {
	$maincanvas delete $connid-notes
	$maincanvas delete $connid-notes-shadow
    }
    #if we got here from undo, record this action as undoable
    if [llength $args] {
	global undoList redoList
	lappend undoList [list "sub_connection" $connid]
	# new actions invalidate the redo list
	set redoList ""
    }
}

proc shadow { pos } {
    return [list [expr 1+[lindex $pos 0]] [expr 1+[lindex $pos 1]]]
}

proc scalePath { path } {
    set minipath ""
    global SCALEX SCALEY
    set doingX 1
    foreach point $path {
	if $doingX { lappend minipath [expr round($point/$SCALEX)] 
	} else { lappend minipath [expr round($point/$SCALEY)] }
	set doingX [expr !$doingX]
    }
    return $minipath
}
    
proc getConnectionNotesOptions { id } {
    set path [eval routeConnection [parseConnectionID $id]]
    set off 7
    # set true if input module is left of the output module on the canvas
    set left [expr [lindex $path 0] > [lindex $path end-1]]
    switch [llength $path] {
	4 { 
	    return [list [lindex $path 0] \
			[expr ([lindex $path 1]+[lindex $path 3])/2-$off] \
		    -width 0 -anchor s]
	}
	8 { 
	    # if output module is right of input module
	    set x1 [expr $left?[lindex $path 6]:[expr [lindex $path 0]+$off]]
	    set x2 [expr $left?[expr [lindex $path 0]-$off]:[lindex $path 6]]
	    return [list [expr $x1+($x2-$x1)/2] [expr [lindex $path 3]-$off] \
			-width [expr $x2-$x1] -anchor s]
	}	
	default {
	    set x [expr ($left?[lindex $path 4]:[lindex $path 6])+$off]
	    set x2 [expr ($left?[lindex $path 2]:[lindex $path 8])-2*$off]
	    set y [expr ($left?[expr [lindex $path 3]-$off]:\
			       [expr [lindex $path end-2]+$off])]
	    return [list $x $y -width [expr $x2-$x] \
			-anchor [expr $left?"sw":"nw"]]
	}
    }
}

proc getModuleNotesOptions { module } {
    global maincanvas NotesPos
    set bbox [$maincanvas bbox $module]
    set off 2
    switch $NotesPos($module) {
	n {
	    return [list [lindex $bbox 0] [lindex $bbox 1] \
			-anchor sw -justify left]
	}
	s {
	    return [list [lindex $bbox 0] [lindex $bbox 3]  \
			-anchor nw -justify left]
	}
	w {
	    return [list [expr [lindex $bbox 0] - $off] [lindex $bbox 1] \
			-anchor ne -justify left]
	}
	# east is default
	default {
	    return [list [expr [lindex $bbox 2] + $off] [lindex $bbox 1] \
			-anchor nw -justify right]
	}

    }
}


proc rebuildConnections { list } {
    global maincanvas minicanvas modname_font
    foreach conn $list {
	set id [lindex $conn 0]
	set path [eval routeConnection [lrange $conn 1 end]]
	eval $maincanvas coords $id $path
	eval $minicanvas coords $id [scalePath $path]
	drawNotes $id
    }
}

proc startPortConnection {modid which porttype portname x y} {
    global maincanvas modname_font new_conn_ports potential_connection
    set isoport [string equal $porttype o]
    set oppositeporttype [expr $isoport?"i":"o"]
    $maincanvas create text [computePortCoords $modid $which $isoport] \
	-text "$portname" -font $modname_font -tags "tempname" \
	-fill white -anchor [expr $isoport?"nw":"sw"]
    set new_conn_ports [netedit find.$oppositeporttype.ports $modid $which]
    foreach i $new_conn_ports {
	if $isoport { set path [eval routeConnection $modid $which $i]
	} else { set path [eval routeConnection $i $modid $which] }
	eval $maincanvas create line $path -width 2 \
	    -tags \"tempconnections [join "temp $i" ""]\"
    }
    set potential_connection ""
}

proc trackPortConnection {modid which porttype x y} {
    global new_conn_ports maincanvas potential_connection Color
    if ![llength $new_conn_ports] return
    set isoport [string equal $porttype o]
    set ox1 [winfo x $maincanvas.module$modid.port$porttype$which]
    set ox2 [lindex [$maincanvas coords $modid] 0]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $maincanvas.module$modid.port$porttype$which]
    set oy2 [lindex [$maincanvas coords $modid] 1]
    set y [expr $y+$oy1+$oy2]
    set c [computePortCoords $modid $which $isoport]
    set mindist [eval computeDist $x $y $c]
    set minport ""
    foreach i $new_conn_ports {
	set c [eval computePortCoords $i [expr !$isoport]]
	set dist [eval computeDist $x $y $c]
	if {$dist < $mindist} {
	    set mindist $dist
	    set minport $i
	}
    }
    $maincanvas itemconfigure tempconnections -fill black
    set potential_connection ""
    if {$minport != ""} {	
	$maincanvas raise [join "temp $minport" ""]
	$maincanvas itemconfigure [join "temp $minport" ""] -fill $Color(Trace)
	if {$isoport} { set potential_connection "$modid $which $minport"
	} else { set potential_connection "$minport $modid $which" }
    } 
}

proc endPortConnection {} {
    global maincanvas potential_connection
    $maincanvas delete tempname
    $maincanvas delete tempconnections
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
    set outpos [computePortCoords $omodid $owhich 1]
    set inpos [computePortCoords $imodid $iwhich 0]
    set ox [lindex $outpos 0]
    set oy [lindex $outpos 1]
    set ix [lindex $inpos 0]
    set iy [lindex $inpos 1]
    if {$ox == $ix && $oy < $iy} {
	return [list $ox $oy $ix $iy]
    } elseif {[expr $oy+19] < $iy} {
	set my [expr ($oy+$iy)/2]
	return [list $ox $oy $ox $my $ix $my $ix $iy]
    } else {
	set mx [expr ($ox<$ix?$ox:$ix)-50]
	return [list $ox $oy $ox [expr $oy+10] $mx [expr $oy+10] \
		    $mx [expr $iy-10] $ix [expr $iy-10] $ix $iy]
    }
}

proc computePortCoords {modid which isoport} {
    global maincanvas port_spacing port_width
    set at [$maincanvas coords $modid]
    set x [expr $which*$port_spacing+6+$port_width/2+[lindex $at 0]]
    set h [winfo height $maincanvas.module$modid]
    # this is to get rid of a bug for modules not mapped to the canvas
    set h [expr $h>1?$h:57]
    set y [expr ($isoport?$h:0) + [lindex $at 1]]
    return [list $x $y]
}

proc computeDist {x1 y1 x2 y2} {
    set dx [expr $x2-$x1]
    set dy [expr $y2-$y1]
    return [expr sqrt($dx*$dx+$dy*$dy)]
}

global ignoreModuleMove 
set ignoreModuleMove 1

proc moduleStartDrag {maincanvas modid x y toggleOnly} {
    global ignoreModuleMove CurrentlySelectedModules rebuildConnectionList
    set ignoreModuleMove 0

    if $toggleOnly {
	$modid toggleSelected $maincanvas 1
	set ignoreModuleMove 1
	return
    }

    #raise the module icon
    set wname "$maincanvas.module$modid"
    raise $wname

    #set module movement coordinates
    global startX startY lastX lastY
    set lastX $x
    set lastY $y
    set startX $x
    set startY $y
       
    #if clicked module isnt selected, unselect all and select this
    if { ![$modid is_selected] } { $modid toggleSelected $maincanvas 0 } 

    #build a connection list for all selected modules to draw pipes when moving
    set rebuildConnectionList ""
    foreach csm $CurrentlySelectedModules {
	eval lappend rebuildConnectionList [netedit getconnected $csm]
    }
    
    #create a gray bounding box around moving modules
    if {[llength $CurrentlySelectedModules] > 1} {
	$maincanvas create rectangle [compute_bbox $maincanvas] \
	    -outline darkgray -tags tempbox
    }
}

proc moduleDrag {maincanvas minicanvas modid x y} {
    global ignoreModuleMove CurrentlySelectedModules rebuildConnectionList
    if $ignoreModuleMove return
    global grouplastX grouplastY lastX lastY
    set bbox [compute_bbox $maincanvas]
    # When the user tries to drag a group of modules off the canvas,
    # Offset the lastX and or lastY variable, so that they can only drag
    #groups to the border of the canvas
    set min_possibleX [expr [lindex $bbox 0] + $x - $lastX]
    set min_possibleY [expr [lindex $bbox 1] + $y - $lastY]    
    if {$min_possibleX <= 0} { set lastX [expr [lindex $bbox 0] + $x] } 
    if {$min_possibleY <= 0} { set lastY [expr [lindex $bbox 1] + $y] }
    set max_possibleX [expr [lindex $bbox 2] + $x - $lastX]
    set max_possibleY [expr [lindex $bbox 3] + $y - $lastY]
    if {$max_possibleX >= 4500} { set lastX [expr [lindex $bbox 2]+$x-4500] }
    if {$max_possibleY >= 4500} { set lastY [expr [lindex $bbox 3]+$y-4500] }

    # Move each module individually
    foreach csm $CurrentlySelectedModules {
	do_moduleDrag $maincanvas $minicanvas $csm $x $y
    }	
    set lastX $grouplastX
    set lastY $grouplastY
    # redraw connections between moved modules
    rebuildConnections [lsort -unique $rebuildConnectionList]
    # move the bounding selection rectangle
    $maincanvas coords tempbox [compute_bbox $maincanvas]
}    

proc do_moduleDrag {maincanvas minicanvas modid x y} {
    global lastX lastY grouplastX grouplastY SCALEX SCALEY
    global mainCanvasWidth mainCanvasHeight

    set grouplastX $x
    set grouplastY $y
    set bbox [$maincanvas bbox $modid]
    
    # Canvas Window width and height
    set width  [winfo width  $maincanvas]
    set height [winfo height $maincanvas]

    # Total Canvas width and height
    set canWidth  [expr double($mainCanvasWidth)]
    set canHeight [expr double($mainCanvasHeight)]
        
    # Cursor movement delta from last position
    set dx [expr $x - $lastX]
    set dy [expr $y - $lastY]

    # if user attempts to drag module off left edge of canvas
    set modx [lindex $bbox 0]
    set left [$maincanvas canvasx 0] 
    if { [expr $modx+$dx] <= $left } {
	if { $left > 0 } {
	    $maincanvas xview moveto [expr ($modx+$dx)/$canWidth]
	}
	if { [expr $modx+$dx] <= 0 } {
	    $maincanvas move $modid [expr -$modx] 0
	    $minicanvas move $modid [expr (-$modx)/$SCALEX] 0
	    set dx 0
	}
    }
    
    #if user attempts to drag module off right edge of canvas
    set modx [lindex $bbox 2]
    set right [$maincanvas canvasx $width] 
    if { [expr $modx+$dx] >= $right } {
	if { $right < $canWidth } {
	    $maincanvas xview moveto [expr ($modx+$dx-$width)/$canWidth]
	}
	if { [expr $modx+$dx] >= $canWidth } {
	    $maincanvas move $modid [expr $canWidth-$modx] 0
	    $minicanvas move $modid [expr ($canWidth-$modx)/$SCALEX] 0
	    set dx 0
	} 
    }
    
    #if user attempts to drag module off top edge of canvas
    set mody [lindex $bbox 1]
    set top [$maincanvas canvasy 0]
    if { [expr $mody+$dy] <= $top } {
	if { $top > 0 } {
	    $maincanvas yview moveto [expr ($mody+$dy)/$canHeight]
	}    
	if { [expr $mody+$dy] <= 0 } {
	    $maincanvas move $modid 0 [expr -$mody]
	    $minicanvas move $modid 0 [expr (-$mody)/$SCALEY]
	    set dy 0
	}
    }
 
    #if user attempts to drag module off bottom edge of canvas
    set mody [lindex $bbox 3]
    set bottom [$maincanvas canvasy $height]
    if { [expr $mody+$dy] >= $bottom } {
	if { $bottom < $canHeight } {
	    $maincanvas yview moveto [expr ($mody+$dy-$height)/$canHeight]
	}	
	if { [expr $mody+$dy] >= $canHeight } {
	    $maincanvas move $modid 0 [expr $canHeight-$mody]
	    $minicanvas move $modid 0 [expr ($canHeight-$mody)/$SCALEY]
	    set dy 0
	}
    }

    # X and Y coordinates of canvas origin
    set Xbounds [winfo rootx $maincanvas]
    set Ybounds [winfo rooty $maincanvas]
    set currx [expr $x-$Xbounds]

    #cursor-boundary check and warp for x-axis
    if { [expr $x-$Xbounds] > $width } {
	cursor warp $maincanvas $width [expr $y-$Ybounds]
	set currx $width
	set scrollwidth [.bot.neteditFrame.vscroll cget -width]
	set grouplastX [expr $Xbounds + $width - 5 - $scrollwidth]
    }
    if { [expr $x-$Xbounds] < 0 } {
	cursor warp $maincanvas 0 [expr $y-$Ybounds]
	set currx 0
	set grouplastX $Xbounds
    }
    
    #cursor-boundary check and warp for y-axis
    if { [expr $y-$Ybounds] > $height } {
	cursor warp $maincanvas $currx $height
	set scrollwidth [.bot.neteditFrame.hscroll cget -width]
	set grouplastY [expr $Ybounds + $height - 5 - $scrollwidth]
    }
    if { [expr $y-$Ybounds] < 0 } {
	cursor warp $maincanvas $currx 0
	set grouplastY $Ybounds
    }
    
    # if there is no movement to perform, then return
    if {!$dx && !$dy} { return }
    
    # Perform the actual move of the module window
    $maincanvas move $modid $dx $dy
    $minicanvas move $modid [expr $dx / $SCALEX ] [expr $dy / $SCALEY ]
    
    drawNotes $modid
}


proc drawNotes { args } {
    global Color Notes Font modname_font maincanvas NotesPos HelpText
    set Font(Notes) $modname_font
    foreach id $args {
	if { ![info exists NotesPos($id)] } {
	    set NotesPos($id) def
	}
	
	set isModuleNotes [winfo exists $maincanvas.module$id]

	if {$NotesPos($id) == "tooltip"} {
	    if { $isModuleNotes } {
		Tooltip $maincanvas.module$id $Notes($id) 
	    } else {
		canvasTooltip $id $Notes($id)
	    }
	} else {
	    if { $isModuleNotes } {
		Tooltip $maincanvas.module$id $HelpText(Module)
	    } else {
		canvasTooltip $id $HelpText(Connection)
	    }
	}

	if { $NotesPos($id) == "none" || $NotesPos($id) == "tooltip"} {
	    $maincanvas delete $id-notes
	    $maincanvas delete $id-notes-shadow
	    continue
	}
	
	if { ![info exists Color(Notes-$id)] } { 
	    if { [info exists Color($id)] } {
		set Color(Notes-$id) $Color($id)
	    } else {
		set Color(Notes-$id) white 
	    }
	}

	if { ![info exists Notes($id)] } { set Notes($id) "" }
	
	if { ![canvasExists $id-notes] } {
	    $maincanvas create text 0 0 -text "" \
		-tags "$id-notes notes" -fill white
	    $maincanvas create text 0 0 -text "" -fill black \
		-tags "$id-notes-shadow shadow"
	}

        set shadowCol [expr [brightness $Color(Notes-$id)]>0.2?"black":"white"]

	if { [canvasExists $id-notes] } {	
	    if { $isModuleNotes } {
		set opt [getModuleNotesOptions $id]
	    } else {
		set opt [getConnectionNotesOptions $id]
	    }
	    $maincanvas coords $id-notes [lrange $opt 0 1]
	    $maincanvas coords $id-notes-shadow [shadow [lrange $opt 0 1]]    
	    eval $maincanvas itemconfigure $id-notes [lrange $opt 2 end]
	    eval $maincanvas itemconfigure $id-notes-shadow [lrange $opt 2 end]
	    $maincanvas itemconfigure $id-notes	-fill $Color(Notes-$id) \
		-font $Font(Notes) -text "$Notes($id)"
	    $maincanvas itemconfigure $id-notes-shadow -fill $shadowCol \
		-font $Font(Notes) -text "$Notes($id)"
		
	    if {!$isModuleNotes} {
		$maincanvas bind $id-notes <ButtonPress-1> \
		    "notesWindow $id notesDoneConnection"
		$maincanvas bind $id-notes <ButtonPress-2> \
		    "global NotesPos; set NotesPos($id) none; drawNotes $id"
	    } else {
		$maincanvas bind $id-notes <ButtonPress-1> \
		    "notesWindow $id notesDoneModule"
		$maincanvas bind $id-notes <ButtonPress-2> \
		    "global NotesPos; set NotesPos($id) none; drawNotes $id"
	    }
	    canvasTooltip $id-notes $HelpText(Notes)		
	}
    }
    $maincanvas raise shadow
    $maincanvas raise notes
}
    


proc moduleEndDrag {mframe maincanvas x y} {
    global ignoreModuleMove CurrentlySelectedModules startX startY
    if $ignoreModuleMove return
    $maincanvas delete tempbox
    computeModulesBbox
    # If only one module was selected and moved, then unselect when done
    if {([expr abs($startX-$x)] > 2 || [expr abs($startY-$y)] > 2) && \
	    [llength $CurrentlySelectedModules] == 1} unselectAll    
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


global Box
set Box(InitiallySelected) ""
set Box(x0) 0
set Box(y0) 0

proc startBox {X Y maincanvas keepselected} {
    global Box CurrentlySelectedModules
    set Box(InitiallySelected) $CurrentlySelectedModules
    if {!$keepselected} {
	unselectAll
	set Box(InitiallySelected) ""
    }
    #Canvas Relative current X and Y positions
    set Box(x0) [expr $X - [winfo rootx $maincanvas] + [$maincanvas canvasx 0]]
    set Box(y0) [expr $Y - [winfo rooty $maincanvas] + [$maincanvas canvasy 0]]
    # Create the bounding box graphic
    $maincanvas create rectangle $Box(x0) $Box(y0) $Box(x0) $Box(y0)\
	-tags "tempbox temp"    
}

proc makeBox {X Y maincanvas} {    
    global Box CurrentlySelectedModules
    #Canvas Relative current X and Y positions
    set x1 [expr $X - [winfo rootx $maincanvas] + [$maincanvas canvasx 0]]
    set y1 [expr $Y - [winfo rooty $maincanvas] + [$maincanvas canvasy 0]]
    #redraw box
    $maincanvas coords tempbox $Box(x0) $Box(y0) $x1 $y1
    # select all modules which overlap the current bounding box
    set overlappingModules ""
    set overlap [$maincanvas find overlapping $Box(x0) $Box(y0) $x1 $y1]
    foreach i $overlap {
	set s "[$maincanvas itemcget $i -tags]"
	if {[$maincanvas type $s] == "window"} {
	    lappend overlappingModules $s
	    if { ![$s is_selected] } {
		$s addSelected $maincanvas
	    }
	}
    }
    # remove those not initally selected or overlapped by box
    foreach mod $CurrentlySelectedModules {
	if {[lsearch $overlappingModules $mod] == -1 && \
		[lsearch $Box(InitiallySelected) $mod] == -1} {
	    $mod removeSelected $maincanvas
	}
    }
}

proc endBox {X Y maincanvas} {
    $maincanvas delete tempbox
}

proc unselectAll {} {
    global CurrentlySelectedModules maincanvas
    foreach i $CurrentlySelectedModules {
	$i removeSelected $maincanvas
    }
}

# Courtesy of the Tcl'ers Wiki (http://mini.net/tcl)
proc brightness { color } {
    foreach {r g b} [winfo rgb . $color] break
    set max [lindex [winfo rgb . white] 0]
    expr {($r*0.3 + $g*0.59 + $b*0.11)/$max}
 } ;#RS, after [Kevin Kenny]

proc blend { c1 c2 } {
    foreach {r1 g1 b1} [winfo rgb . $c1] break
    foreach {r2 g2 b2} [winfo rgb . $c2] break
    set max [expr double([lindex [winfo rgb . white] 0])]
    set r [expr int(((($r1/$max)+($r2/$max))/2)*255)]
    set g [expr int(((($g1/$max)+($g2/$max))/2)*255)]
    set b [expr int(((($b1/$max)+($b2/$max))/2)*255)]
    return [format "\#%02x%02x%02x" $r $g $b]
 } 