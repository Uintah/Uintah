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

global CurrentlySelectedModules
set CurrentlySelectedModules ""

global startX
set startX 0

global startY
set startY 0

global undoList
set undoList ""

global redoList
set redoList ""

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
    public isSubnetModule 0
    public subnetNumber 0

    method compiling_p {} { return $compiling_p }
    method set_compiling_p { val } { 
	set compiling_p $val	
	setColorAndTitle
        if {[winfo exists .standalone]} {
	    app indicate_dynamic_compile [modname] [expr $val?"start":"stop"]
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
    method make_icon {modx mody} {
	global $this-done_bld_icon Disabled Subnet Color
	set $this-done_bld_icon 0
	set Disabled([modname]) 0
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	
	set modframe $canvas.module[modname]
	frame $modframe -relief raised -borderwidth 3 
	
	bind $modframe <1> "moduleStartDrag [modname] %X %Y 0"
	bind $modframe <2> "createSubnet $Subnet([modname])"	    
	bind $modframe <B1-Motion> "moduleDrag [modname] %X %Y"
	bind $modframe <ButtonRelease-1> "moduleEndDrag [modname] %X %Y"
	bind $modframe <Control-Button-1> "moduleStartDrag [modname] %X %Y 1"
	bind $modframe <3> "moduleMenu %X %Y [modname]"
	
	frame $modframe.ff
	set p $modframe.ff
	pack $p -side top -expand yes -fill both -padx 5 -pady 6

	global ui_font modname_font time_font
	if {[have_ui]} {
	    button $p.ui -text "UI" -borderwidth 2 -font $ui_font \
		-anchor center -command "$this initialize_ui"
	    pack $p.ui -side left -ipadx 5 -ipady 2
	}
	# Make the Subnet Button
	if {$isSubnetModule} {
	    button $p.subnet -text "Sub-Network" -borderwidth 2 \
		-font $ui_font -anchor center \
		-command "showSubnetWindow $subnetNumber"
	    pack $p.subnet -side bottom -ipadx 5 -ipady 2
	}

	# Make the title
	label $p.title -text "$name" -font $modname_font -anchor w
	pack $p.title -side [expr $make_progress_graph?"top":"left"] \
	    -padx 2 -anchor w
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
	    -tags "module [modname]" -anchor nw

	global SCALEX SCALEY minicanvas
	$minicanvas create rectangle [expr $modx/$SCALEX] [expr $mody/$SCALEY]\
		[expr $modx/$SCALEX + 4] [expr $mody/$SCALEY + 2] \
		-outline "" -fill $Color(Basecolor) -tags "module [modname]"

	# Create, draw, and bind all input and output ports
	drawPorts [modname]
	
	# Try to find a position for the icon where it doesn't
	# overlap other icons
	set done 0
	set modx [expr int($modx)]
	set mody [expr int($mody)]
	while { $done == 0 } {
	    set x1 $modx
	    set y1 $mody
	    set x2 [expr $modx+120]
	    set y2 [expr $mody+50]
	    
	    set tagIDs [$canvas find overlapping $x1 $y1 $x2 $y2]
	    set done 0
	    foreach tagID $tagIDs {
		set tags [$canvas gettags $tagID]
		set pos [lsearch $tags module]
		set isMod [lsearch $tags [modname]]
		if { $pos == -1 && $isMod == -1} {
		    set done 1
		    break
		}
	    }
		
	    if {!$done} {
		$canvas move [modname] 0 80
		$minicanvas move [modname] 0 [expr 80 / $SCALEY ]
		incr mody 80
		set canbot [$canvas canvasy [winfo height $canvas]]
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
	
	$this setColorAndTitle
	update idletasks
    }
    
    method setColorAndTitle {args} {
	global Subnet Color Disabled
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set m $canvas.module[modname]
	set color $Color(Basecolor)
	if { [$this is_selected] } { set color $Color(Selected) }
	if { $Disabled([modname])} { set color [blend $color $Color(Disabled)]}
	if { $compiling_p } {
	    set color $Color(Compiling)
	    set args "COMPILING"
	}
	if { ![llength $args] && ![string first SubnetIcon [modname]] } {
	    set args $Subnet(Subnet$Subnet([modname]_num)_Name)
	}
	$m configure -background $color
	$m.ff configure -background $color
	$m.ff.title configure -background $color
	if {[$this have_ui]} { $m.ff.ui configure -background $color }
	if {$make_time} { $m.ff.time configure -background $color }
	if {$isSubnetModule} { $m.ff.subnet configure -background $color }
	if {![llength $args]} { set args $name }
	if {![llength $args]} { set args $name }
	$m.ff.title configure -text "$args" -justify left
	update idletasks
    }
       
    method addSelected {} {
	if {![$this is_selected]} { 
	    global CurrentlySelectedModules
	    lappend CurrentlySelectedModules [modname]
	    setColorAndTitle
	}
    }    

    method removeSelected {} {
	if {[$this is_selected]} {
	    #Remove me from the Currently Selected Module List
	    global CurrentlySelectedModules
	    listFindAndRemove CurrentlySelectedModules [modname]
	    setColorAndTitle
	}
    }
    
    method toggleSelected { option } {
	if { $option == 0 } { unselectAll }
	if [is_selected] { removeSelected
	} else { addSelected }
    }
    
    method update_progress {} {
	set width [expr int($progress*($graph_width-4))]
	if {!$make_progress_graph || $width == $old_width } return
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set graph $canvas.module[modname].ff.inset.graph
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
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	$canvas.module[modname].ff.time configure -text $tstr
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
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	$canvas.module[modname].ff.inset.graph configure -background $color
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
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set indicator $canvas.module[modname].ff.msg.indicator
	place forget $indicator
	$indicator configure -width $indicator_width -background $color
	place $indicator -relheight 1 -anchor nw 

	if {[winfo exists .standalone]} {
	    app indicate_error [modname] $msg_state
	}
	
    }

    method get_x {} {
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	return [lindex [$canvas coords [modname]] 0]
    }

    method get_y {} {
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	return [lindex [$canvas coords [modname]] 1]
    }

    method is_selected {} {
	global CurrentlySelectedModules
	return [expr ([lsearch $CurrentlySelectedModules [modname]]!=-1)?1:0]
    }

    method displayLog {} {
	if [$this is_subnet] return
	set w .mLogWnd[modname]
	
	# does the window exist?
	if [winfo exists $w] {
	    raise $w
	    return
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
	# TODO: unregister only for streams with the supplied output
	button $w.fbuttons.ok -text "OK" \
	    -command "$this destroyStreamOutput $w"
	
	pack $w.log $w.fbuttons -side top -padx 5 -pady 5
	pack $w.fbuttons.ok -side right -padx 5 -pady 5 -ipadx 3 -ipady 3

	$msgLogStream registerOutput $w.log.txt
    }
    
    method destroyStreamOutput {w} {
	# TODO: unregister only for streams with the supplied output
	$msgLogStream unregisterOutput
	destroy $w
    }


    method resize {} {
	set iports [expr [llength [getModulePortinfo [modname] i]]+1]
	set oports [expr [llength [getModulePortinfo [modname] o]]+1]
	set ports [expr $oports>$iports?$oports:$iports]
	if {[set $this-done_bld_icon]} {
	    global Subnet port_spacing
	    set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	    set width [expr 8+$ports*$port_spacing] 
	    set width [expr ($width < $initial_width)?$initial_width:$width]
	    $canvas itemconfigure [modname] -width $width
	}
    }

    method module_grow { args } {
	$this resize
    }
    
    method module_shrink {} {
	$this resize
    }

    method setDone {} {
	#module actually mapped to the canvas
	if {[set $this-done_bld_icon] == 0 } {
	    set $this-done_bld_icon 1	
	    global Subnet
	    set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	    set initial_width [winfo width $canvas.module[modname]]
	    module_grow
	}
    }
    method lightOPort { which color } {
	lightPort [list [modname] $which o] $color
    }
    method lightIPort { which color } {
	lightPort [list [modname] $which i] $color
    }

    method execute {} {
	$this-c needexecute
    }
    
    method is_subnet {} {
	return $isSubnetModule
    }
	
}   

proc moduleMenu {x y modid} {
    global Subnet mouseX mouseY
    set mouseX $x
    set mouseY $y
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)
    set menu_id "$canvas.module$modid.ff.menu"
    regenModuleMenu $modid $menu_id
    tk_popup $menu_id $x $y    
}

proc regenModuleMenu {modid menu_id} {
    # Wipe the menu clean...
    for {set c 0} {$c <= 10 } {incr c } {
	$menu_id delete $c
    }
    global Subnet Disabled mouseX mouseY
    set name [$modid name]
    $menu_id add command -label "$modid" -state disabled
    $menu_id add separator
    $menu_id add command -label "Execute" -command "$modid execute"
    if {![$modid is_subnet]} {
	$menu_id add command -label "Help" -command "moduleHelp $modid"
    }
    $menu_id add command -label "Notes" \
	-command "notesWindow $Subnet($modid) $modid notesDoneModule"
    if [$modid is_selected] { 
	$menu_id add command -label "Destroy Selected" \
	    -command "moduleDestroySelected"
    }
    $menu_id add command -label "Destroy" -command "moduleDestroy $modid"
    if {![$modid is_subnet]} {
	$menu_id add command -label "Show Log" -command "$modid displayLog"
    }
    $menu_id add command -label "Make Subnet" \
	-command "createSubnet $Subnet($modid)"

    if ![llength $Subnet(${modid}_connections)] return
    if $Disabled($modid) {
	$menu_id add command -label "Enable" -command "disableModule $modid 0"
    } else {
	$menu_id add command -label "Disable" -command "disableModule $modid 1"
    }
}

# args == { connid omodid owhich imodid iwhich }
proc connectionMenu {x y conn} {
    global Subnet
    set canvas $Subnet(Subnet$Subnet([lindex $conn 0])_canvas)
    set menu_id "$canvas.menu[makeConnID $conn]"
    regenConnectionMenu $menu_id $conn
    tk_popup $menu_id $x $y
}

proc regenConnectionMenu { menu_id conn } {
    # create menu if it doesnt exist
    if ![winfo exists $menu_id] {
	menu $menu_id -tearoff 0 -disabledforeground white
    }
    # Wipe the menu clean...
    for {set c 0} {$c <= 10 } {incr c } {
	$menu_id delete $c
    }
    global Subnet Disabled
    $menu_id add command -label "Connection" -state disabled
    $menu_id add separator
    $menu_id add command -label "Delete" -command "destroyConnection {$conn}"
    set connid [makeConnID $conn]
    set label [expr $Disabled($connid)?"Enable":"Disable"]
    $menu_id add command -command "disableConnection {$conn}" -label $label
    set subnet $Subnet([lindex $conn 0])
    $menu_id add command -label "Notes" -command "notesWindow $subnet $connid"
}

proc notesDoneModule { id } {
    global Notes $id-notes
    set $id-notes $Notes($id)
}

proc notesWindow { subnet id {done ""} } {
    global Notes Color NotesPos Subnet
    if { [winfo exists .notes] } { destroy .notes }
    toplevel .notes
    text .notes.input -relief sunken -bd 2 -height 20
    frame .notes.b
    button .notes.b.done -text "Done" \
	-command "okNotesWindow $subnet $id \"$done\""
    button .notes.b.clear -text "Clear" -command ".notes.input delete 1.0 end"
    button .notes.b.cancel -text "Cancel" -command "destroy .notes"
    set rgb white
    if { [info exists Color($id)] } { set rgb $Color($id) }
    button .notes.b.reset -fg black -text "Reset Color" -command \
	"set Color(Notes-$id) $rgb; .notes.b.color configure -bg $rgb"

    if { [info exists Color(Notes-$id)] } { set rgb $Color(Notes-$id) }
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

proc okNotesWindow {subnet id {done  ""}} {
    global Notes
    set Notes($id) [.notes.input get 1.0 "end - 1 chars"]
    destroy .notes
    if { $done != ""} { eval $done $id }
    drawNotes $subnet $id
    update idletasks
}

proc disableModule { module state } {
    global Disabled CurrentlySelectedModules
    set mods [expr [$module is_selected]?"$CurrentlySelectedModules":"$module"]
    foreach modid $mods { ;# Iterate through the modules
	foreach conn [getModuleConnections $modid] { ;# all module connections
	    if { $Disabled([makeConnID $conn]) != $state } {
		disableConnection $conn
	    }
	}
    }
}

proc checkForDisabledModules { args } {
    global Disabled
    set args [lsort -unique $args]
    foreach modid $args {
	# assume module is disabled
	set Disabled($modid) 1
	foreach conn [getModuleConnections $modid] {
	    # if connection is enabled, then enable module
	    if { !$Disabled([makeConnID $conn]) } {
		set Disabled($modid) 0
		# module is enabled, continue onto next module
		break;
	    }
	}
	$modid setColorAndTitle
    }
}

proc makeConnID { args } {
    while {[llength $args] == 1} { set args [lindex $args 0] }
    set args [lreplace $args 1 1 p[oNum args]]
    set args [lreplace $args 3 3 p[iNum args]]
    set args [linsert $args 2 to]
    return [join $args "_"]
} 

proc drawConnections { connlist } {
    global Color Disabled HelpText Subnet
    foreach conn $connlist {
	set id [makeConnID $conn]
	set path [routeConnection $conn]
	set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
	set minicanvas $Subnet(Subnet$Subnet([oMod conn])_minicanvas)
	if { ![canvasExists $canvas $id] } {
	    eval $canvas create bline $path -tags $id
	    eval $minicanvas create line [scalePath $path] -tags $id
	    $canvas bind $id <1> "$canvas raise $id; TraceConnection {$conn}"
	    $canvas bind $id <Control-Button-2> "destroyConnection {$conn}"
	    $canvas bind $id <3> "connectionMenu %X %Y {$conn}"
	    $canvas bind $id <ButtonRelease> "+deleteTrace {$conn}"
	    canvasTooltip $id $HelpText(Connection)
	} else {
	    eval $canvas coords $id $path
	    eval $minicanvas coords $id [scalePath $path]
	}
	update idletasks
	set color [expr $Disabled($id)?"$Color(ConnDisabled)":"$Color($id)"]
	set wid   [expr $Disabled($id)?3:7]
	if {$color == ""} { set color red }
	$canvas itemconfigure $id -width $wid -borderwidth 2 -fill "$color"
	$minicanvas itemconfigure $id -width 1 -fill "$color"
	$minicanvas lower $id
	drawNotes $Subnet([oMod conn]) $id
    }
}


# Deletes red connections on canvas and turns port lights black
# conn = { omodid owhich imodid iwhich }
proc deleteTrace { conn } {
    global Subnet Color TracedPort
    set TracedPorts ""
    $Subnet(Subnet$Subnet([oMod conn])_canvas) delete temp
    $Subnet(Subnet$Subnet([oMod conn])_minicanvas) delete temp
    $Subnet(Subnet$Subnet([oMod conn])_minicanvas) itemconfigure module \
	-fill $Color(Basecolor)
    lightPort [oPort conn] black
    lightPort [iPort conn] black
}


global TracedPorts
set TracedPorts ""

proc TracePort { port } {
    global TracedPorts
    if { [lsearch $TracedPorts $port] != -1} return
    lappend TracedPorts port
    foreach conn [getModuleConnections [pMod port]] {
	# if 1. Port not already traced and
	#    2. Connection originates at this module and
	#    3. Either we want to trace all module ports or
	#       $conn is the port we want to trace
	if { [string equal [pMod port] [[pType port]Mod conn]] && \
	     [pNum port] == [[pType port]Num conn] } {
	    lappend TracedPorts [oPort port] [iPort port]
	    lightConnection $conn
	    TraceAll [[invType port]Mod conn] [pType port]
	}
    }
}

proc TraceAll { modid porttype } {
    global TracedPorts
    set invtype [expr [string equal o $porttype]?"i":"o"]
    foreach conn [getModuleConnections $modid] {
	if { [string equal [${porttype}Mod conn] $modid] } {
	    lightConnection $conn
	    TraceAll [${invtype}Mod conn] $porttype
	}
    }
}


proc TraceConnection { conn } {
    global TracedPorts Color
    set TracedPorts ""
    lightPort [oPort conn] $Color(Trace)
    lightPort [iPort conn] $Color(Trace)
    lightConnection $conn
    TracePort [oPort conn]
    TracePort [iPort conn]
}

proc canvasExists { canvas arg } {
    return [expr [llength [$canvas find withtag $arg]]?1:0]
}

proc disableConnection { conn } {
    global Subnet Disabled Color
    set connid [makeConnID $conn]
    if {!$Disabled($connid)} {
	set Disabled($connid) 1
	netedit blockconnection $connid
	set Color(Notes-$connid) $Color(ConnDisabled)
    } else {
	set Disabled($connid) 0
	netedit unblockconnection $connid
	set Color(Notes-$connid) $Color($connid)
    }
    $Subnet(Subnet$Subnet([oMod conn])_canvas) raise $connid
    drawConnections [list $conn]
    checkForDisabledModules [oMod conn] [iMod conn]
}

proc lightConnection { conn } {
    global Subnet Color
    set path [routeConnection $conn]
    set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
    eval $canvas create bline $path \
	-width 7 -borderwidth 2 -fill $Color(Trace) -tags temp
    $canvas raise temp

    set minicanvas $Subnet(Subnet$Subnet([oMod conn])_minicanvas)
    eval $minicanvas create line [scalePath $path] -width 1 \
	-fill $Color(Trace) -tags temp
    $minicanvas itemconfigure [oMod conn] -fill green
    $minicanvas itemconfigure [iMod conn] -fill green
    $minicanvas raise temp
}


proc addConnection { omodid owhich imodid iwhich } {
    createConnection [list $omodid $owhich $imodid $iwhich]
}


# set the optional args command to anything to record the undo action
proc createConnection { conn { undo 0 } { tell_SCIRun 1 } } {
    global Subnet Notes Disabled Color
    if { ![info exists Subnet([oMod conn])] || \
	     ![info exists Subnet([iMod conn])] || \
	     $Subnet([oMod conn]) != $Subnet([iMod conn]) } {
	puts "Not creating connection $conn: Subnet levels dont match"
	return
    }
    
    # Trying to create subnet connections on the main network editor window
    # most likely the user is loading a subnet into the main window
    if {($Subnet([oMod conn]) == 0 && [isaSubnetEditor [oMod conn]]) || \
	($Subnet([iMod conn]) == 0 && [isaSubnetEditor [iMod conn]]) } {
	return
    }

    if { $tell_SCIRun} {
	# Traverse the subnet levels to find the real connecting ports
	set realConn [findRealConnection $conn]
	if { ![isaSubnet [oMod realConn]] && ![isaSubnet [iMod realConn]] } {
	    # if the modules at both ends are not subnet ports
	    # tell SCIRun to create this connection
	    if {[eval netedit addconnection $realConn] == ""} {
		tk_messageBox -type ok -parent . -icon warning -message \
		    "Cannot create connection:\n addConnection $conn." 
		return
	    }	    
	}
    }
    

    lappend Subnet([oMod conn]_connections) $conn
    lappend Subnet([iMod conn]_connections) $conn
    set connid [makeConnID $conn]
    if ![info exists Notes($connid)] { set Notes($connid) "" }
    if ![info exists Disabled($connid)] { set Disabled($connid) 0 }
    if ![info exists Color($connid)] {
	set Color($connid) \
	    [lindex [lindex [getModulePortinfo [oMod conn] o] [oNum conn]] 0]
	if ![llength $Color($connid)] { set $Color($connid) red }
    }

    drawConnections [list $conn]
    drawPorts [oMod conn] o
    drawPorts [iMod conn] i

    #if we got here from undo, record this action as undoable
    if { $undo } {
	global undoList redoList
	lappend undoList [list "createConnection" $conn]
	# new actions invalidate the redo list
	set redoList ""	
    }
    update idletasks
}
		      


proc destroyConnection { conn { undo 0 } { tell_SCIRun 1 } } { 
    if { $tell_SCIRun } {
	# Traverse the subnet levels to find the real connecting ports
	set realConn [findRealConnection $conn]
	if { ![isaSubnet [oMod realConn]] && ![isaSubnet [iMod realConn]] } {
	    # if the modules at both ends are not subnet ports
	    # tell SCIRun to delete this connection
	    netedit deleteconnection [makeConnID $realConn]
	}
    }

    global Subnet Disabled Color
    listFindAndRemove Subnet([oMod conn]_connections) $conn
    listFindAndRemove Subnet([iMod conn]_connections) $conn
    set connid [makeConnID $conn]
    $Subnet(Subnet$Subnet([oMod conn])_canvas) delete \
	$connid $connid-notes $connid-notes-shadow
    $Subnet(Subnet$Subnet([oMod conn])_minicanvas) delete \
	$connid $connid-notes $connid-notes-shadow
    array unset Disabled $connid
    array unset Color $connid
    drawPorts [oMod conn] o
    drawPorts [iMod conn] i

    #if we got here from undo, record this action as undoable
    if { $undo } {
	global undoList redoList
	lappend undoList [list "destroyConnection" $conn]
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
    set path [routeConnection [parseConnectionID $id]]
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
    global Subnet NotesPos
    set bbox [$Subnet(Subnet$Subnet($module)_canvas) bbox $module]
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

proc startPortConnection { port portname} {
    global Subnet modname_font new_conn_ports potential_connection
    set potential_connection ""
    set subnet $Subnet([pMod port])
    set canvas $Subnet(Subnet${subnet}_canvas)
    set isoport [string equal [pType port] o]
    $canvas create text [computePortCoords $port] \
	-text "$portname" -font $modname_font -tags temp \
	-fill white -anchor [expr $isoport?"nw":"sw"]
    set new_conn_ports ""
    foreach newport [findPorts $port] {
	set newport [findPortTermination "$newport [invType port]"]
	if { $Subnet([pMod newport]) == $subnet } {
	    lappend new_conn_ports $newport
	}
    }
    # if the subnet is not level 0 (meaning the main network editor)
    # create a subnet input or output port for the module
    # and the port is an output port or input port that is not used
    if { $subnet && ($isoport || [llength $new_conn_ports]) } {
	set addSubnetPort 1
	foreach conn $Subnet([pMod port]_connections) {
	    if { [pNum port] == [[pType port]Num conn] && \
		     [string equal Subnet$subnet [[invType port]Mod conn]]} {
		set addSubnetPort 0
		break
	    }
	}
	if {$addSubnetPort} {
	    lappend new_conn_ports \
		[list Subnet$subnet [numPorts $subnet [invType port]]]
	}
    }

    foreach i $new_conn_ports {
	if $isoport { set path [routeConnection "[pMod port] [pNum port] $i"]
	} else { set path [routeConnection "$i [pMod port] [pNum port]"] }
	eval $canvas create line $path -width 2 \
	    -tags \"temp tempconnections [join "temp $i" ""]\"
    }
}

proc trackPortConnection { port x y } {
    global new_conn_ports potential_connection Color Subnet
    if ![llength $new_conn_ports] return
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set isoport [string equal [pType port] o]
    set ox1 [winfo x $canvas.module[pMod port].port[pType port][pNum port]]
    set ox2 [lindex [$canvas coords [pMod port]] 0]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $canvas.module[pMod port].port[pType port][pNum port]]
    set oy2 [lindex [$canvas coords [pMod port]] 1]
    set y [expr $y+$oy1+$oy2]
    set c [computePortCoords $port]
    set mindist [eval computeDist $x $y $c]
    set minport ""
    foreach i $new_conn_ports {
	set c [computePortCoords "$i [invType port]"]
	set dist [eval computeDist $x $y $c]
	if {$dist < $mindist} {
	    set mindist $dist
	    set minport $i
	}
    }
    $canvas itemconfigure tempconnections -fill black
    set potential_connection ""
    if {$minport != ""} {
	$canvas raise [join "temp $minport" ""]
	$canvas itemconfigure [join "temp $minport" ""] -fill $Color(Trace)
	if {$isoport} { set potential_connection "[pMod port] [pNum port] $minport"
	} else { set potential_connection "$minport [pMod port] [pNum port]" }
    } 
}

proc endPortConnection { subnet } {
    global Subnet potential_connection
    $Subnet(Subnet${subnet}_canvas) delete temp
    if { $potential_connection == "" } return
    createConnection $potential_connection 1
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
    listRemove undoList end
    # Add it to the redo list
    lappend redoList $undo_item

    set action [lindex $undo_item 0]

    if { $action == "createConnection" } {
	destroyConnection [lindex $undo_item 1]
    }
    if { $action == "destroyConnection" } {
	createConnection [lindex $undo_item 1]
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
    listRemove redoList end
    # Add it to the undo list
    lappend undoList $redo_item

    eval $redo_item
}

proc routeConnection { conn } {
    set outpos [computePortCoords [oPort conn]]
    set inpos [computePortCoords [iPort conn]]
    set ox [lindex $outpos 0]
    set oy [lindex $outpos 1]
    set ix [lindex $inpos 0]
    set iy [lindex $inpos 1]
    if {$ox == $ix && $oy <= $iy} {
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

proc computePortCoords { port } {
    global Subnet port_spacing port_width
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set isoport [string equal o [pType port]]

    if { [string first "Subnet" [pMod port]] == 0 &&
	 [string first "SubnetIcon" [pMod port]] != 0} {
	if { !$isoport } { 
	    set at [list [$canvas canvasx 0] \
			[$canvas canvasy [winfo height $canvas]]]
	} else {
	    set at [list [$canvas canvasx 0] \
			[$canvas canvasy 0]]
	}
	set h 0
    } elseif { [lsearch $Subnet(Subnet$Subnet([pMod port])_Modules) [pMod port]]!= -1} {
	set at [$canvas coords [pMod port]]
	set h [winfo height $canvas.module[pMod port]]
	# this is to get rid of a bug for modules not mapped to the canvas
	set h [expr $h>1?$h:57]
    } else {
	return [list 0 0]
    }
    
    set x [expr [pNum port]*$port_spacing+6+$port_width/2+[lindex $at 0]]
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

proc moduleStartDrag {modid x y toggleOnly} {
    global ignoreModuleMove CurrentlySelectedModules redrawConnectionList
    set ignoreModuleMove 0
    if $toggleOnly {
	$modid toggleSelected 1
	set ignoreModuleMove 1
	return
    }

    global Subnet startX startY lastX lastY
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)

    #raise the module icon
    raise $canvas.module$modid

    #set module movement coordinates
    set lastX $x
    set lastY $y
    set startX $x
    set startY $y
       
    #if clicked module isnt selected, unselect all and select this
    if { ![$modid is_selected] } { $modid toggleSelected 0 }

    #build a connection list of all selected modules to draw conns when moving
    set redrawConnectionList ""
    foreach csm $CurrentlySelectedModules {
	eval lappend redrawConnectionList [getModuleConnections $csm]
    }
    set redrawConnectionList [lsort -unique $redrawConnectionList]
    
    #create a gray bounding box around moving modules
    if {[llength $CurrentlySelectedModules] > 1} {
	set bbox [compute_bbox $canvas]
	$canvas create rectangle  $bbox -outline black -tags tempbox
    }
}

proc moduleDrag {modid x y} {
    global ignoreModuleMove CurrentlySelectedModules redrawConnectionList
    if $ignoreModuleMove return
    global Subnet grouplastX grouplastY lastX lastY
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)
    set bbox [compute_bbox $canvas]
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
	do_moduleDrag $csm $x $y
    }	
    set lastX $grouplastX
    set lastY $grouplastY
    # redraw connections between moved modules
    drawConnections $redrawConnectionList
    # move the bounding selection rectangle
    $canvas coords tempbox [compute_bbox $canvas]
}    

proc do_moduleDrag {modid x y} {
    global Subnet minicanvas lastX lastY grouplastX grouplastY SCALEX SCALEY
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)

    set grouplastX $x
    set grouplastY $y
    set bbox [$canvas bbox $modid]
    
    # Canvas Window width and height
    set width  [winfo width  $canvas]
    set height [winfo height $canvas]

    # Total Canvas Scroll Region width and height
    set canScroll [$canvas cget -scrollregion]
    set canWidth  [expr double([lindex $canScroll 2] - [lindex $canScroll 0])]
    set canHeight [expr double([lindex $canScroll 3] - [lindex $canScroll 1])]
        
    # Cursor movement delta from last position
    set dx [expr $x - $lastX]
    set dy [expr $y - $lastY]

    # if user attempts to drag module off left edge of canvas
    set modx [lindex $bbox 0]
    set left [$canvas canvasx 0] 
    if { [expr $modx+$dx] <= $left } {
	if { $left > 0 } {
	    $canvas xview moveto [expr ($modx+$dx)/$canWidth]
	}
	if { [expr $modx+$dx] <= 0 } {
	    $canvas move $modid [expr -$modx] 0
	    $minicanvas move $modid [expr (-$modx)/$SCALEX] 0
	    set dx 0
	}
    }
    
    #if user attempts to drag module off right edge of canvas
    set modx [lindex $bbox 2]
    set right [$canvas canvasx $width] 
    if { [expr $modx+$dx] >= $right } {
	if { $right < $canWidth } {
	    $canvas xview moveto [expr ($modx+$dx-$width)/$canWidth]
	}
	if { [expr $modx+$dx] >= $canWidth } {
	    $canvas move $modid [expr $canWidth-$modx] 0
	    $minicanvas move $modid [expr ($canWidth-$modx)/$SCALEX] 0
	    set dx 0
	} 
    }
    
    #if user attempts to drag module off top edge of canvas
    set mody [lindex $bbox 1]
    set top [$canvas canvasy 0]
    if { [expr $mody+$dy] <= $top } {
	if { $top > 0 } {
	    $canvas yview moveto [expr ($mody+$dy)/$canHeight]
	}    
	if { [expr $mody+$dy] <= 0 } {
	    $canvas move $modid 0 [expr -$mody]
	    $minicanvas move $modid 0 [expr (-$mody)/$SCALEY]
	    set dy 0
	}
    }
 
    #if user attempts to drag module off bottom edge of canvas
    set mody [lindex $bbox 3]
    set bottom [$canvas canvasy $height]
    if { [expr $mody+$dy] >= $bottom } {
	if { $bottom < $canHeight } {
	    $canvas yview moveto [expr ($mody+$dy-$height)/$canHeight]
	}	
	if { [expr $mody+$dy] >= $canHeight } {
	    $canvas move $modid 0 [expr $canHeight-$mody]
	    $minicanvas move $modid 0 [expr ($canHeight-$mody)/$SCALEY]
	    set dy 0
	}
    }

    # X and Y coordinates of canvas origin
    set Xbounds [winfo rootx $canvas]
    set Ybounds [winfo rooty $canvas]
    set currx [expr $x-$Xbounds]

    #cursor-boundary check and warp for x-axis
    if { [expr $x-$Xbounds] > $width } {
	cursor warp $canvas $width [expr $y-$Ybounds]
	set currx $width
	set scrollwidth [.bot.neteditFrame.vscroll cget -width]
	set grouplastX [expr $Xbounds + $width - 5 - $scrollwidth]
    }
    if { [expr $x-$Xbounds] < 0 } {
	cursor warp $canvas 0 [expr $y-$Ybounds]
	set currx 0
	set grouplastX $Xbounds
    }
    
    #cursor-boundary check and warp for y-axis
    if { [expr $y-$Ybounds] > $height } {
	cursor warp $canvas $currx $height
	set scrollwidth [.bot.neteditFrame.hscroll cget -width]
	set grouplastY [expr $Ybounds + $height - 5 - $scrollwidth]
    }
    if { [expr $y-$Ybounds] < 0 } {
	cursor warp $canvas $currx 0
	set grouplastY $Ybounds
    }
    
    # if there is no movement to perform, then return
    if {!$dx && !$dy} { return }
    
    # Perform the actual move of the module window
    $canvas move $modid $dx $dy
    $minicanvas move $modid [expr $dx / $SCALEX ] [expr $dy / $SCALEY ]
    
    drawNotes $Subnet($modid) $modid
}


proc drawNotes { subnet args } {
    global Subnet Color Notes Font NotesPos HelpText modname_font
    set Font(Notes) $modname_font
    set canvas $Subnet(Subnet${subnet}_canvas)
    foreach id $args {
	if { ![info exists NotesPos($id)] } {
	    set NotesPos($id) def
	}
	
	set isModuleNotes \
	    [expr [lsearch $Subnet(Subnet${subnet}_Modules) $id]!=-1?1:0]
	
	if {$NotesPos($id) == "tooltip"} {
	    if { $isModuleNotes } {
		Tooltip $canvas.module$id $Notes($id) 
	    } else {
		canvasTooltip $id $Notes($id)
	    }
	} else {
	    if { $isModuleNotes } {
		Tooltip $canvas.module$id $HelpText(Module)
	    } else {
		canvasTooltip $id $HelpText(Connection)
	    }
	}
	
	if { $NotesPos($id) == "none" || $NotesPos($id) == "tooltip"} {
	    $canvas delete $id-notes $id-notes-shadow
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
	
	if { ![canvasExists $canvas $id-notes] } {
	    $canvas create text 0 0 -text "" \
		-tags "$id-notes notes" -fill white
	    $canvas create text 0 0 -text "" -fill black \
		-tags "$id-notes-shadow shadow"
	}
	
        set shadowCol [expr [brightness $Color(Notes-$id)]>0.2?"black":"white"]
	
	
	if { $isModuleNotes } { 
	    set opt [getModuleNotesOptions $id]
	} else {
	    set opt [getConnectionNotesOptions $id]
	}
	
	$canvas coords $id-notes [lrange $opt 0 1]
	$canvas coords $id-notes-shadow [shadow [lrange $opt 0 1]]    
	eval $canvas itemconfigure $id-notes [lrange $opt 2 end]
	eval $canvas itemconfigure $id-notes-shadow [lrange $opt 2 end]
	$canvas itemconfigure $id-notes	-fill $Color(Notes-$id) \
	    -font $Font(Notes) -text "$Notes($id)"
	$canvas itemconfigure $id-notes-shadow -fill $shadowCol \
	    -font $Font(Notes) -text "$Notes($id)"
	
	if {!$isModuleNotes} {
	    $canvas bind $id-notes <ButtonPress-1> "notesWindow $subnet $id"
	    $canvas bind $id-notes <ButtonPress-2> \
		"global NotesPos;set NotesPos($id) none; drawNotes $subnet $id"
	} else {
	    $canvas bind $id-notes <ButtonPress-1> \
		"notesWindow $subnet $id notesDoneModule"
	    $canvas bind $id-notes <ButtonPress-2> \
		"global NotesPos;set NotesPos($id) none; drawNotes $subnet $id"
	}
	canvasTooltip $id-notes $HelpText(Notes)		
    }
    $canvas raise shadow
    $canvas raise notes
}
    


proc moduleEndDrag {modid x y} {
    global Subnet ignoreModuleMove CurrentlySelectedModules startX startY
    if $ignoreModuleMove return
    $Subnet(Subnet$Subnet($modid)_canvas) delete tempbox
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

proc moduleDestroy {modid} {
    global Subnet

    # Deleting the module connections backwards works for dynamic modules
    set modList [getModuleConnections $modid]
    set size [expr [llength $modList]-1]
    for {set j $size} {$j >= 0} {incr j -1} {
	destroyConnection [lindex $modList $j]
    }

    # Delete Icon from canvases
    $Subnet(Subnet$Subnet($modid)_canvas) delete $modid
    destroy $Subnet(Subnet$Subnet($modid)_canvas).module$modid
    $Subnet(Subnet$Subnet($modid)_minicanvas) delete $modid
    
    # Remove references to module is various state arrays
    array unset Subnet ${modid}_connections
    $modid removeSelected
    listFindAndRemove Subnet(Subnet$Subnet($modid)_Modules) $modid
    $modid delete
    if { ![isaSubnetIcon $modid] } {
	netedit deletemodule $modid
    }
    
    # Kill the modules UI if it exists
    if {[winfo exists .ui$modid]} {
	destroy .ui$modid
    }
}

proc moduleDestroySelected {} {
    global CurrentlySelectedModules 
    foreach mnum $CurrentlySelectedModules { moduleDestroy $mnum }
}


global Box
set Box(InitiallySelected) ""
set Box(x0) 0
set Box(y0) 0

proc startBox {canvas X Y keepselected} {
    global Box CurrentlySelectedModules
    
    set Box(InitiallySelected) $CurrentlySelectedModules
    if {!$keepselected} {
	unselectAll
	set Box(InitiallySelected) ""
    }
    #Canvas Relative current X and Y positions
    set Box(x0) [expr $X - [winfo rootx $canvas] + [$canvas canvasx 0]]
    set Box(y0) [expr $Y - [winfo rooty $canvas] + [$canvas canvasy 0]]
    # Create the bounding box graphic
    $canvas create rectangle $Box(x0) $Box(y0) $Box(x0) $Box(y0)\
	-tags "tempbox temp"    
}

proc makeBox {canvas X Y} {    
    global Box CurrentlySelectedModules
    #Canvas Relative current X and Y positions
    set x1 [expr $X - [winfo rootx $canvas] + [$canvas canvasx 0]]
    set y1 [expr $Y - [winfo rooty $canvas] + [$canvas canvasy 0]]
    #redraw box
    $canvas coords tempbox $Box(x0) $Box(y0) $x1 $y1
    # select all modules which overlap the current bounding box
    set overlappingModules ""
    set overlap [$canvas find overlapping $Box(x0) $Box(y0) $x1 $y1]
    foreach id $overlap {
	set tags [$canvas gettags $id] 
	set pos [lsearch -exact $tags "module"]
	if { $pos != -1 } {
	    set modname [lreplace $tags $pos $pos]
	    lappend overlappingModules $modname
	    if { ![$modname is_selected] } {
		$modname addSelected
	    }
	}
    }
    # remove those not initally selected or overlapped by box
    foreach mod $CurrentlySelectedModules {
	if {[lsearch $overlappingModules $mod] == -1 && \
		[lsearch $Box(InitiallySelected) $mod] == -1} {
	    $mod removeSelected
	}
    }
}

proc unselectAll {} {
    global CurrentlySelectedModules
    foreach i $CurrentlySelectedModules {
	$i removeSelected
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

proc getCanvasModules { canvas } {
    set retval ""
    foreach tagid [$canvas find withtag "module"] {
	set tags [$canvas gettags $tagid]
	set pos  [lsearch -exact $tags "module"]
	lappend retval [lreplace $tags $pos $pos]
    }
    return [lsort -unique $retval]
}


# Returns all connections including subnet connections
proc getModuleConnections { module } {
    global Subnet
    return $Subnet(${module}_connections)
}

proc getModulePortinfo { modid porttype } {
    global Subnet
    if {[string first Subnet $modid] != 0} {
	set retval [$modid-c ${porttype}portinfo] 
    } else {
	set retval ""
	set tocheck ""
	if { [string first SubnetIcon $modid] == 0} {
	    set subnet $Subnet(${modid}_num)
	} else {
	    set subnet $Subnet($modid)
	    set porttype [expr [string equal o $porttype]?"i":"o"]
	}
	set invport [expr [string equal o $porttype]?"i":"o"]
	set connections $Subnet(Subnet${subnet}_connections)
	foreach conn $connections {
	    if { [string equal Subnet${subnet} [${invport}Mod conn]] } {
		lappend tocheck \
		    [list [${invport}Num conn] [${porttype}Port conn]]
	    }	    
	}
	set i 0
	foreach check [lsort -integer -index 0 $tocheck] {
	    while { $i < [lindex $check 0] } {
		lappend retval [list red 0 Bad Port]
		incr i
	    }
	    set mod [lindex [lindex $check 1] 0]
	    set num [lindex [lindex $check 1] 1]
	    lappend retval [lindex [getModulePortinfo $mod $porttype] $num]
	    incr i
	}
    }

    set connectedPortNumbers ""
    if {[info exists Subnet(${modid}_connections)]} {
	foreach conn $Subnet(${modid}_connections) {
	    if { [string equal [${porttype}Mod conn] $modid] } {
		lappend connectedPortNumbers [${porttype}Num conn]
	    }
	}
    }
    set realRetVal ""
    set i 0
    foreach portinfo $retval {	
	if { [lsearch $connectedPortNumbers $i] == -1 } {
	    lappend realRetVal [lreplace $portinfo 1 1 0]
	} else {
	    lappend realRetVal [lreplace $portinfo 1 1 1]
	}
	incr i
    }
    return $realRetVal
}


proc findPorts { port } { 
    set newport [findPortOrigin $port]
    if {![string first Subnet [pMod newport]]} {
	return ""
    }
    set ports [netedit find[invType port]ports [pMod newport] [pNum newport]]
    return $ports
}

proc findPortOrigin { port } {
    global Subnet
    set modified 1
    while { $modified && ![string first Subnet [pMod port]] } {
	set modified 0	
	if { ![string first SubnetIcon [pMod port]] } {
	    # loop through encapsulated Subnet Editors' i/o connections
	    set searchname Subnet$Subnet([pMod port]_num)
	} else { ;# else module is a Subnet Editor Port
	    # loop through the encompasing Subnet Icons' i/o connections
	    set searchname SubnetIcon$Subnet([pMod port])
	}	    
	foreach c $Subnet(${searchname}_connections) {
	    # if connection ends at the right subnet editor output port
	    if {[pNum port] == [[invType port]Num c] && \
		    [string equal $searchname [[invType port]Mod c]] } {
		set port "[[pType port]Mod c] [[pType port]Num c] [pType port]"
		set modified 1
		break
	    }
	}
    }
    return [lrange $port 0 1]
}

proc findPortTermination { port } {
    global Subnet
    set modified 1
    while { $modified } {
	set modified 0	
	foreach c $Subnet([pMod port]_connections) {
	    # if connection ends at the right subnet editor output port
	    if {[pNum port] == [[pType port]Num c] && \
		    [string equal [pMod port] [[pType port]Mod c]]} {
		if {[string first SubnetIcon [[invType port]Mod c]] != 0 && \
			[string first Subnet [[invType port]Mod c]] == 0} {
		    set module SubnetIcon$Subnet([[invType port]Mod c])
		    set port "$module [[invType port]Num c] [pType port]"
		    set modified 1
		    break
		}
	    }
	}
    }
    return [lrange $port 0 1]
}



proc findRealConnection { conn } {
    return "[findPortOrigin [oPort conn]] [findPortOrigin [iPort conn]]"
}

proc drawPorts { modid { porttypes "i o" } } {
    global Subnet port_spacing port_width port_height port_light_height
    set isSubnet [expr ([string first Subnet $modid] == 0) && \
			([string first SubnetIcon $modid] != 0)]
    set subnet $Subnet($modid)
    if {$isSubnet} {
	set modframe .subnet${subnet}.can
    } else {
	set modframe $Subnet(Subnet${subnet}_canvas).module$modid
	$modid resize
    }

    foreach porttype $porttypes {
	set isoport [string equal $porttype o]

	set i 0
	while {[winfo exists $modframe.port$porttype$i]} {
	    destroy $modframe.port$porttype$i
	    destroy $modframe.portlight$porttype$i
	    incr i
	}
	set portinfo [getModulePortinfo $modid $porttype]
	set i 0
	foreach t $portinfo {
	    set portcolor [lindex $t 0]
	    set portname [join [lrange $t 2 3] ":"]
	    set x [expr $i*$port_spacing+6]
	    if { $isSubnet } { set x [expr $x+3 ]}
	    set e [expr $isoport?"bottom":"top"]
	    if { $isSubnet || [lindex $t 1]} { set e out$e }
	    set portbevel $modframe.port$porttype$i
	    set portlight $modframe.portlight$porttype$i
	    bevel $portbevel -width $port_width -height $port_height \
		-borderwidth 3 -edge $e -background $portcolor \
		-pto 2 -pwidth 7 -pborder 2
	    frame $portlight -width $port_width -height 4 \
		-relief raised -background black -borderwidth 0
	    if { $isSubnet } {
		if $isoport {
		    place $portbevel -bordermode outside \
			-y $port_light_height -anchor nw -x $x
		    place $portlight -in $portbevel -x 0 -y 0 -anchor sw
		} else {
		    place $portbevel -bordermode ignore \
			-x $x -rely 1 -y -4 -anchor sw
		    place $portlight -in $portbevel -x 0 -rely 1.0 -anchor nw
		} 
	    } else {
		if $isoport {
		    place $portbevel -bordermode ignore \
			-rely 1 -anchor sw -x $x
		    place $portlight -in $portbevel -x 0 -y 0 -anchor sw
		} else {
		    place $portbevel -bordermode outside -x $x -y 0 -anchor nw
		    place $portlight -in $portbevel -x 0 -rely 1.0 -anchor nw
		}
	    }

	    set port [list $modid $i $porttype]
	    foreach p [list $portbevel $portlight] {
		bind $p <2> "startPortConnection {$port} {$portname}"
		bind $p <B2-Motion> "trackPortConnection {$port} %x %y"
		bind $p <ButtonRelease-2> "endPortConnection $subnet"
		bind $p <ButtonPress-1> "TracePort {$port}"
		bind $p <ButtonRelease-1> "deleteTrace"
	    }
	    
	    global HelpText
	    Tooltip $port $HelpText(Connection)
	    Tooltip $portlight $HelpText(Connection)
	    incr i
	} 
    }
    if $isSubnet {
	drawPorts SubnetIcon$Subnet($modid)
    }
}

proc lightPort { port color } {
    global Subnet
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set p $canvas.module[pMod port].portlight[pType port][pNum port]
    if {[winfo exists $p]} {
	$p configure -background $color
    }
}

proc isaSubnet { modid } {
    return [expr [string first Subnet $modid] == 0]
}

proc isaSubnetIcon { modid } {
    return [expr [string first SubnetIcon $modid] == 0]
}

proc isaSubnetEditor { modid } {
    return [expr ![isaSubnetIcon $modid] && [isaSubnet $modid]]
}


# NOTE: The following functions use upvar!
# This means you call them without expanding the variable beforehand
# Correct Example:
#    set connection { Module1 0 Module2 1 }
#    set outputModule [oMod connection]     <- CORRECT
#    set outputModule [oMod $connection]    <- INCORRECT
#    set connname connection
#    set outputModule [oMod $connname]	    <- CORRECT 


# Returns the Module ID of the port
proc pMod { port_varname } {
    upvar $port_varname port
    return [lindex $port 0]
}

# Returns the Number of the port
proc pNum { port_varname } {
    upvar $port_varname port
    return [lindex $port 1]
}

# Returns the Type of the port 'i' for Input or 'o' for Output
proc pType { port_varname } {
    upvar $port_varname port
    return [lindex $port 2]
}

# Returns the Inverse Type of the port 'o' for Input or 'i' for Output
proc invType { port_varname } {
    upvar $port_varname port    
    return [expr [string equal o [lindex $port 2]]?"i":"o"]
}

# Returns the Output Module of the connection
proc oMod { connection_varname } {
    upvar $connection_varname connection
    return [lindex $connection 0]
}

# Returns the Output Port Number of the connection
proc oNum { connection_varname } {
    upvar $connection_varname connection
    return [lindex $connection 1]
}

# Returns the Output Port of the connection
proc oPort { connection_varname } {
    upvar $connection_varname connection
    return "[lrange $connection 0 1] o"
}

# Returns the Input Module of the connection
proc iMod { connection_varname } {
    upvar $connection_varname connection
    return [lindex $connection 2]
}

# Returns the Input Port Number of the connection
proc iNum { connection_varname } {
    upvar $connection_varname connection
    return [lindex $connection 3]
}

# Returns the Input Port of the connection
proc iPort { connection_varname } {
    upvar $connection_varname connection
    return "[lrange $connection 2 3] i"
}
