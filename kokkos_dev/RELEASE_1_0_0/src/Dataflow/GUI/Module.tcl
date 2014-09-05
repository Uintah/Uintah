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

set port_spacing 18
set port_width 13
set port_height 7

global selected_color
global unselected_color

global connection_list
set connection_list ""

set selected_color darkgray
set unselected_color gray

global MModuleFakeConnections
set MModuleFakeConnections ""

global CurrentlySelectedModules
set CurrentlySelectedModules ""

global CurrentMacroModules
set CurrentMacroModules ""

global MacroedModules
set MacroedModules ""

global modules
set modules ""

global font_pixel_width
set font_pixel_width 0.0

global extra_ports
set extra_ports 0

global original_size
set original_title_size 0


itcl_class Module {
   
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
			
    constructor {config} {
	set msgLogStream [TclStream msgLogStream#auto]
        global $this-notes
	
	if [info exists $this-notes] {
	    set dum 0
	} else {
	    set $this-notes ""
	}
	
	# messages should be accumulating
	if {[info exists $this-msgStream]} {
	    $msgLogStream registerVar $this-msgStream
	} else {
	    puts "No stream buffer variable exists"
	}

	set MacroModule ""
	set Macroed 0
	set menumod 0
    }

    destructor {
	set w .mLogWnd[modname]
	if {[winfo exists $w]!=0} {
	    destroy $w
	}
	
	$msgLogStream destructor
	destroy $this
    }
    
    method config {config} {
    }

    public msgLogStream
    public name
    protected canvases ""
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_width 0
    protected mdragged
    protected mconnected
    protected last_pos
    protected MacroModule
    protected Macroed
    protected menumod
    protected MenuList
    protected made_icon 0
    public state "NeedData" {$this update_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}
    public group -1
    public selected 0
    public show_status 1
    method name {} {
	return name
    }
    
    method get_group {} {
	global $this-group
	return [eval $$this-group]
    }

    method set_last_pos { lp } {
	set last_pos $lp
    }

    method get_last_pos {} {
	return $last_pos
    }
    
    method Macroed {} {
	return $Macroed
    }
    
    method MacroModule {} {
	if { $MacroModule == "" } {
	    return [modname]
	}
	return $MacroModule
    }
 
    method set_Macroed { m } {
	set Macroed $m
    }
    
    method set_MacroModule { mm } {
	set MacroModule $mm
    }
   

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


    method get_oports { which } {
	set mmodid [$this MacroModule]
	if { [string match [$mmodid mod_type] "module"] == 1 } {
	    return "{$mmodid $which}"
	} else {
	    foreach opm [$mmodid get_oport_mapping] {
		if { [string match [lindex $opm 0] [$this modname]] == 1 } {
		    if { [string match [lindex $opm 1] $which] == 1 } {
			return "{$mmodid [lindex $opm 2]}"
		    }
		}
	    }
	}
	# routine should never get this far...
    }

    method get_iports { which } {
	set mmodid [$this MacroModule]
	
	if { [string match [$mmodid mod_type] "module"] == 1 } {
	    return "{$mmodid $which}"
	} else {
	    foreach ipm [$mmodid get_iport_mapping] {
		if { [string match [lindex $ipm 0] [$this modname]] == 1 } {
		    if { [string match [lindex $ipm 1] $which] == 1 } {
			return "{$mmodid [lindex $ipm 2]}"
		    }
		}
	    }
	}
	# routine should never get this far...
    }




    #  Make the modules icon on a particular canvas
    method make_icon {canvas minicanvas modx mody} {
	global modules
	set modules "$modules [modname]"
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

	# Make the title
	label $p.title -text $name -font $modname_font -anchor w
	pack $p.title -side top -padx 2 -anchor w

	# Make the time label
	if {$make_time} {
	    label $p.time -text "00.00" -font $time_font
	    pack $p.time -side left -padx 2
	}

	# Make the progress graph
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

# DMW: commenting this out for now, since MacroModules can't yet be saved as
#   part of a net
#	$p.menu add command -label "Group Selected" -command "makeMacroModule\
#		$canvas $minicanvas [modname]"

	$p.menu add command -label "Destroy Selected" \
		-command "moduleDestroySelected $canvas $minicanvas $this"
	$p.menu add command -label "Destroy" \
		-command "moduleDestroy $canvas $minicanvas [modname]"
	$p.menu add command -label "Show Log" \
		-command "$this displayLog"

	global $this-show_status
	$p.menu add checkbutton -variable $this-show_status -label\
		"Show Status"

	set $this-show_status 1

# Destroy selected items with a Ctrl-D press
	bind all <Control-d> "moduleDestroySelected \
		$canvas $minicanvas lightgray"
# Clear the canvas
	bind all <Control-l> "ClearCanvas"

        
# Select the clicked item, and unselect all others
	bind $p <2> "$this toggleSelected $canvas 0"
	bind $p.title <2> "$this toggleSelected $canvas 0"
	if {$make_time} {
	    bind $p.time <2> "$this toggleSelected $canvas 0"
	    bind $p.inset <2> "$this toggleSelected $canvas 0" 
	}
	if {[$this info method ui] != ""} {
	    bind $p.ui <2> "$this toggleSelected $canvas 0"
	}

# Select the item in focus, leaving all others selected
	bind $p <Control-Button-2> "$this toggleSelected $canvas 1"
	bind $p.title <Control-Button-2> "$this toggleSelected $canvas 1"
	if {$make_time} {
	    bind $p.time <Control-Button-2> "$this toggleSelected $canvas 1"
	    bind $p.inset <Control-Button-2> "$this toggleSelected $canvas 1"
	}
	if {[$this info method ui] != ""} {
	    bind $p.ui <Control-Button-2> "$this toggleSelected $canvas 1"
	}
	 
# Select the item in focus, and unselect all others
	bind .bot.neteditFrame.canvas <2> "startBox %X %Y $canvas 0"
	bind .bot.neteditFrame.canvas <Control-Button-2> "startBox %X %Y $canvas 1"
	bind .bot.neteditFrame.canvas <B2-Motion> "makeBox %X %Y $canvas"
	bind .bot.neteditFrame.canvas <ButtonRelease-2> "endBox %X %Y $canvas"

	bind $p <1> "$canvas raise $this"
	bind $p <1> "moduleStartDrag $canvas [modname] %X %Y"
	bind $p <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $p <ButtonRelease-1> "moduleEndDrag $modframe $canvas"
	bind $p <3> "popup_menu %X %Y $canvas $minicanvas [modname]"
	bind $p.title <1> "moduleStartDrag $canvas [modname] %X %Y"
	bind $p.title <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	bind $p.title <ButtonRelease-1> "moduleEndDrag $modframe $canvas"
	bind $p.title <3> "popup_menu %X %Y $canvas $minicanvas [modname]"
	
	if {$make_time} {
	    bind $p.time <1> "moduleStartDrag $canvas [modname] %X %Y"
	    bind $p.time <B1-Motion> \
		    "moduleDrag $canvas $minicanvas [modname] %X %Y"
	    bind $p.time <ButtonRelease-1> "moduleEndDrag $modframe $canvas"
	    bind $p.time <3> "popup_menu %X %Y $canvas $minicanvas [modname]"
	}
	if {$make_progress_graph} {
	    bind $p.inset <1> "moduleStartDrag $canvas [modname] %X %Y"
	    bind $p.inset <B1-Motion> "moduleDrag $canvas $minicanvas [modname] %X %Y"
	    bind $p.inset <ButtonRelease-1> "moduleEndDrag $modframe $canvas"
	    bind $p.inset <3> "popup_menu %X %Y $canvas $minicanvas [modname]"
	}
	set made_icon 1
    }
    method set_moduleDragged {  ModuleDragged } {
	set mdragged $ModuleDragged
    }
    method get_moduleDragged {} {
	return $mdragged
    }
    method set_moduleConnected { ModuleConnected } {
	set mconnected $ModuleConnected
    }
    method get_moduleConnected {} {
	return $mconnected
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
	rebuildConnections [netedit getconnected [modname]] 0
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
	rebuildConnections [netedit getconnected [modname]] 0
    }
   
    method addSelected {canvas color} {

	global CurrentlySelectedModules
	set m [modname]
	

	if { [string match "*$m*" $CurrentlySelectedModules] == 0 } { 
	    #Add me to the Currently Selected Module 
	    set p $canvas.module[modname].ff
	    
	    $p configure -background $color
	    if {[$this info method ui] != ""} {
		$p.ui configure -background $color
	    }
	    $p.title configure -background $color
	    $canvas.module[modname] configure -background $color
	    
	    if {$make_time} {
		$p.time configure -background $color
	    }	
	    
	    
	    set CurrentlySelectedModules "$CurrentlySelectedModules [modname]"
	    $this sel 1
	    
	}
    }
    

    method removeSelected {canvas color} {
	#Remove me from the Curren2tly Selected Module List
	global CurrentlySelectedModules
	set p $canvas.module[modname].ff
	
	set tempList ""
    
	foreach item $CurrentlySelectedModules {
	    if { $item != [modname] } {
		set tempList "$tempList $item"
	    }
	}
	
	set CurrentlySelectedModules $tempList
	
	$p configure -background $color
	
	if {[$this info method ui] != ""} {
	    $p.ui configure -background $color
	}
	$p.title configure -background $color
	$canvas.module[modname] configure -background $color
	if { [$this info protected make_time -value] == 1 } {
	    $p.time configure -background $color
	}     
	$this sel 0
	
    }
    
    method sel {status} {
	set selected $status
    }

    method mod_type {} {
	return module
    }

    method toggleSelected { canvas option } {
	global CurrentlySelectedModules
	global selected_color
	global unselected_color
	
	if { $option == 0 } {
	    foreach i $CurrentlySelectedModules {
		$i removeSelected $canvas $unselected_color
	    }
	}


	set p $canvas.module[modname].ff
	
	###Add a module to the selected module list
	
	
	if { $selected == 0 } {
	    # Add me to the Currently Selected Module List
	    set color $selected_color
	    addSelected $canvas $color
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
	# call update_progress
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

    method get_this {} {
	return $this
    }

     method get_this_c {} {
	return $this-c
    }

    method is_selected {} {
	global CurrentlySelectedModules
	foreach csm $CurrentlySelectedModules {
	    if { [string match [$this modname] $csm] } {
		return 1
	    }
	}

	return 0
    }

    method menu_modified {} {
	return $menumod
    }

    method set_menu_modified { status } {
	if { $status == 0 || $status == 1 } {
	    set menumod $status
	}
    }

    method get_menu_modified {} {
	return $menumod
    }

    method set_orig_menu { orig_menu } {
	set MenuList $orig_menu
    }
    
    method get_orig_menu {} {
	return $MenuList
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
	set t "$t -- pid=[set $this-pid]"
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
    method module_grow {ports} {  
	global maincanvas
	global font_pixel_width
	global extra_ports
	global original_title_size
	global modname_font
	global port_spacing

	set temp_spacing [expr $port_spacing+1]
	set num_ports $ports
	set mod_width [winfo width $maincanvas.module[modname] ]

	#initialize all values first time through
	if {$original_title_size == 0} {
	    set original_title_size [winfo width $maincanvas.module[modname].ff.title]
	    set font_pixel_width [font measure $modname_font\
		    "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"]
	    set font_pixel_width [expr $font_pixel_width/53.0]
	}
	
	# determine if it needs more room
	if { [expr $mod_width-[expr $num_ports*$temp_spacing] ] <= $temp_spacing } {
	    incr extra_ports 1
	    set title_width [expr $original_title_size+[expr $temp_spacing*$extra_ports]]
	    set title_width [expr int([expr ceil([expr $title_width/$font_pixel_width])])]
	    $maincanvas.module[modname].ff.title configure -width $title_width
	}
    }

    method module_shrink {ports} {  
	global maincanvas
	global font_pixel_width
	global extra_ports
	global original_title_size
	global port_spacing

	set temp_spacing [expr $port_spacing+1]
	set num_ports $ports
	set mod_width [winfo width $maincanvas.module[modname] ]
	set title_width [winfo width $maincanvas.module[modname].ff.title]
	
	#make sure it doesn't shrink smaller than original size
	if { [expr $extra_ports-1] >=0 } {
	    set extra_ports [expr $extra_ports-1]
	}

	#determine if you have enough room to take a port size off
	if { [expr $title_width-$temp_spacing] > $original_title_size } {
	    set title_width [expr $original_title_size+[expr $extra_ports*$temp_spacing]]
	    set title_width [expr $title_width/$font_pixel_width]
	    set title_width [expr int([expr ceil($title_width)])]
	    $maincanvas.module[modname].ff.title configure -width $title_width
	}
    }
}   
proc popup_menu {x y canvas minicanvas modid} {
    global CurrentlySelectedModules
    
    global menu_id

    set menu_id "$canvas.module$modid.ff.menu"

    # What to do if the module is not selected
    
    if { ! [$modid is_selected] } {
	# Get rid of the group modules stuff
	foreach cx [commandIndex $menu_id 1] {
	    if { [string match [lindex $cx 1] "makeMacroModule"] == 1 } {
		$menu_id delete [lindex $cx 0]
	    }
	}
	
	# Get rid of the delete modules stuff
	foreach cx [commandIndex $menu_id 1] {
	    if { [string match [lindex $cx 1] "moduleDestroySelected"] == 1 } {
		$menu_id delete [lindex $cx 0]
	    }
	}
	
	$modid set_menu_modified 1
    }
 
    # What to do if the module is selected
    
    # Regenerate the menu...
    if { [$modid is_selected] } {
	if { [$modid get_menu_modified] == 1 } {
	    regenMenu $modid $menu_id $canvas $minicanvas
	}
    }
        
    # popup the menu
    tk_popup $menu_id $x $y    
}

proc regenMenu {modid menu_id canvas minicanvas} {
    # Wipe the menu clean...
    for {set c 0} {$c <= 10 } {incr c } {
	$menu_id delete $c
    }
    
    $menu_id add command -label "[$modid get_this]" -state disabled
    $menu_id add separator
    $menu_id add command -label "Execute" -command\
	    "[$modid get_this_c] needexecute"
    $menu_id add command -label "Help" -command "moduleHelp [$modid name]"
    $menu_id add command -label "Notes" -command "moduleNotes\
	    [$modid name] [$modid modname]"
    $menu_id add command -label "Group Selected" -command "makeMacroModule\
	    $canvas $minicanvas [$modid modname]"
    $menu_id add command -label "Destroy Selected" \
	    -command "moduleDestroySelected $canvas $minicanvas\
	    [$modid get_this]"
    $menu_id add command -label "Destroy" \
	    -command "moduleDestroy $canvas $minicanvas [$modid modname]"
    global [$modid get_this]-show_status
    $menu_id add checkbutton -variable [$modid get_this]-show_status -label\
	    "Show Status"

    set [$modid get_this]-show_status 1
}

proc commandIndex {menu_id mode} {
    set commandIndex ""

    if { $mode == 1 } {
	for {set c 2} {$c < 10} {incr c} {
	    set command [$menu_id entrycget $c -command]
	    if { [string match $command ""] } {
		break
	    } else {
		set commandIndex "$commandIndex {$c $command}"
	    }
	}
    }
    
    return $commandIndex
    
}



proc startIPortConnection {imodid iwhich x y} {
    global mm
    set mm 0
    set temp_list ""


    # If module is of type macro, save its false connection info,
    # and copy the names and portnumbers of the real ports into
    # imodid and iwhich

    if { [string match [$imodid mod_type] macromodule] == 1} {
	
	# A flag to let us know we're dealing with a macromodule
	set mm 1
	
	set fake_imodid $imodid
	set fake_iwhich $iwhich

	set timodid [lindex [lindex [$imodid get_iport_mapping] $iwhich] 0]
	set tiwhich [lindex [lindex [$imodid get_iport_mapping] $iwhich] 1]
    
	set imodid $timodid
	set iwhich $tiwhich

    }




    # Find all of the OPorts of the same type and draw a temporary line
    # to them....

    global conn_oports
    
    
    set conn_oports [netedit findoports $imodid $iwhich]

    foreach pt $conn_oports {
	set m [lindex $pt 0]
	set ip ""
	set in_port ""

	if { [$m Macroed] == 1} {
	
	    set mmodule [$m MacroModule]
	    set ip [lindex $pt 1]

	    foreach pmap [$mmodule get_oport_mapping] {
		if { [string match $m [lindex $pmap 0]] == 1} {
		    if { [string match $ip [lindex $pmap 1]] ==1} {
			set in_port [lindex $pmap 2]
		    }
		}
	    }
	    
	    set temp_list "$temp_list {$mmodule $in_port}"
	} else {
	    set temp_list "$temp_list {$m [lindex $pt 1]}"
	}
    }
    global new_conn_oports
    set new_conn_oports $temp_list

    global netedit_canvas
        
    if { $mm == 1 } {
	set coords [computeIPortCoords $fake_imodid $fake_iwhich]
    } else {
	set coords [computeIPortCoords $imodid $iwhich]
    }



    set typename [lindex [lindex [$imodid-c iportinfo] $iwhich] 2]
    set portname [lindex [lindex [$imodid-c iportinfo] $iwhich] 3]
    set fullname $typename:$portname

    frame $netedit_canvas.frame
    label $netedit_canvas.frame.label -text $fullname -foreground white -bg #036
    pack $netedit_canvas.frame $netedit_canvas.frame.label    
    $netedit_canvas create window [lindex $coords 0]\
	    [lindex $coords 1] -window $netedit_canvas.frame \
	    -anchor sw -tags "tempname" 
    
#    set curtext [eval $netedit_canvas create text [lindex $coords 0]\
#	    [lindex $coords 1] -anchor sw -text {$fullname} -tags "tempname"]
#    $curtext configure -foreground white
    foreach i $new_conn_oports {
	set omodid [lindex $i 0]
	set owhich [lindex $i 1]
	
	if { $mm == 1 } {
	    set path [join [routeConnection $omodid $owhich \
		    $fake_imodid $fake_iwhich]]
	} else {
	    set path [join [routeConnection $omodid $owhich $imodid $iwhich]]
	}

	eval $netedit_canvas create line $path -width 2 \
	    -tags \"tempconnections iconn$owhich$omodid\"
    }
    global potential_connection
    set potential_connection ""
}

#"


proc startOPortConnection {omodid owhich x y} {
    # Find all of the IPorts of the same type and draw a temporary line
    # to them....

    global mm
    set mm 0

    
    set temp_list ""

    # If module is of type macro, save its false connection info,
    # and copy the names and portnumbers of the real ports into
    # omodid and owhich

    if { [string match [$omodid mod_type] macromodule] == 1} {
	# A flag to let us know we're dealing with a macromodule...
	set mm 1

	set fake_omodid $omodid
	set fake_owhich $owhich

	set tomodid [lindex [lindex [$omodid get_oport_mapping] $owhich] 0]
	set towhich [lindex [lindex [$omodid get_oport_mapping] $owhich] 1]

	set omodid $tomodid
	set owhich $towhich
    	
    }


    global conn_iports
    set conn_iports [netedit findiports $omodid $owhich]
    

    foreach pt $conn_iports {
	set m [lindex $pt 0]
	set op ""
	set out_port ""
	
	if { [$m Macroed] == 1 } {
	    set mmodule [$m MacroModule]
	    set op [lindex $pt 1]

	    foreach pmap [$mmodule get_iport_mapping] {
		if { [string match $m [lindex $pmap 0]] == 1} {
		    if { [string match $op [lindex $pmap 1]] } {
			set out_port [lindex $pmap 2]
		    }
		}
	    }
	    
	    set temp_list "$temp_list {$mmodule $out_port}"
	    
	} else {
	    set temp_list "$temp_list {$m [lindex $pt 1]}"
	}
    }
    global new_conn_iports
    set new_conn_iports $temp_list
    

    global netedit_canvas

    if { $mm == 1 } {
	set coords [computeOPortCoords $fake_omodid $fake_owhich]
    } else {
	set coords [computeOPortCoords $omodid $owhich]
    }

    set typename [lindex [lindex [$omodid-c oportinfo] $owhich] 2]
    set portname [lindex [lindex [$omodid-c oportinfo] $owhich] 3]
    set fullname $typename:$portname
    frame $netedit_canvas.frame
    label $netedit_canvas.frame.label -text $fullname -foreground white -bg #036
    pack $netedit_canvas.frame $netedit_canvas.frame.label    
    $netedit_canvas create window [lindex $coords 0]\
	    [lindex $coords 1] -window $netedit_canvas.frame \
	    -anchor nw -tags "tempname" 
    
#    eval $netedit_canvas create text [lindex $coords 0] [lindex $coords 1] \
#	    -anchor nw -text {$fullname} -tags "tempname"
    foreach i $new_conn_iports {
	set imodid [lindex $i 0]
	set iwhich [lindex $i 1]
	
	if { $mm == 1 } {
	    set path [join [routeConnection\
		    $fake_omodid $fake_owhich $imodid $iwhich]]
	} else {
	    set path [join [routeConnection $omodid $owhich $imodid $iwhich]]
	}

	eval $netedit_canvas create line $path -width 2 \
	    -tags \"tempconnections oconn$iwhich$imodid\"
    }
    global potential_connection
    set potential_connection ""
}

#"



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

#"

proc destroyConnection {connid omodid imodid} {
    global connection_list

    if { [string match [$omodid mod_type] "macromodule"] == 1 &&\
	    [string match [$imodid mod_type] "macromodule"] == 1 } {
	# Find MacroModule Port Numbers

	# omodid port information
	
	for {set ctr 0} {$ctr <= {eval $ctr+1}} {incr ctr} {
	    set b 0
	    if { [string match "MacroModule?_p$ctr*" $connid] == 1} {
		set mmod_oport_num $ctr
		set b 1
	    }
	    if { $b == 1 } {
		break
	    }
	}

	set oport_mapping_info [$omodid get_oport_mapping]

	foreach pmi $oport_mapping_info {
	    set t 0
	    if { [string match [lindex $pmi 2] $mmod_oport_num] == 1} {
		set orci $pmi
		set t 1
	    }
	    if {$t == 1} {
		break
	    }
	}

	set real_omodid [lindex $orci 0]
	set real_owhich [lindex $orci 1]

	# imodid port information
			
	for {set ctr 0} {$ctr <= 1000} {incr ctr} {
	    set b 0
	    if { [string match "*_to_MacroModule?_p$ctr*" $connid] == 1} {
		set mmod_iport_num $ctr
		set b 1
	    }
	    if {$b == 1} {
		break
	    }
	    
	}

	set iport_mapping_info [$imodid get_iport_mapping]

	
	foreach pmi $iport_mapping_info {
	    set t 0
	    if { [string match [lindex $pmi 2] $mmod_iport_num] == 1 } {
		set irci $pmi
		set t 1
	    }
	    if {$t == 1} {
		break
	    }
	}

	set real_imodid [lindex $irci 0]
	set real_iwhich [lindex $irci 1]
	
	set real_connid "$real_omodid"
	append real_connid "_p$real_owhich"
	append real_connid "_to_$real_imodid"
	append real_connid "_p$real_iwhich"
	
	
	set templist ""
	
	foreach con $connection_list {
	    set t 0
	    if { [string match [lindex $con 0] $real_omodid] } {
		if { [string match [lindex $con 1] $real_owhich] } {
		    if { [string match [lindex $con 2] $real_imodid] } {
			if { [string match [lindex $con 3] $real_iwhich] } {
			    set t 1
			}
		    }
		}
	    }
	    
	    if { $t == 0 } {
		set templist "$templist {$con}"
	    }
	}

	set connection_list $templist
	
	netedit deleteconnection $real_connid $real_omodid $real_imodid
	global netedit_canvas netedit_mini_canvas
	$netedit_canvas delete $connid
	$netedit_mini_canvas delete $connid
	configureOPorts $omodid
	configureIPorts $imodid

	set templist ""
	foreach fc [$omodid get_FakeConnections] {
	    if { [string match $connid [lindex $fc 0]] == 0 } {
		set templist "$templist {$fc}"
	    }
	}
	$omodid set_FakeConnections $templist
	

	set templist ""
	foreach fc [$imodid get_FakeConnections] {
	    if { [string match $connid [lindex $fc 0]] == 0 } {
		set templist "$templist {$fc}"
	    }
	}
	$imodid set_FakeConnections $templist

    } elseif { [string match [$omodid mod_type] "macromodule"] == 1 } {

	# A little trick to find the macromodule port number that we're dealing with

	for {set ctr 0} {$ctr <= {eval $ctr+1}} {incr ctr} {
	    set b 0
	    if { [string match "MacroModule?_p$ctr*" $connid] == 1} {
		set mmod_port_num $ctr
		set b 1
	    }
	    if {$b == 1} {
		break
	    }
	}
    	
	set port_mapping_info [$omodid get_oport_mapping]
		
	foreach pmi $port_mapping_info {
	    set t 0
	    if { [string match [lindex $pmi 2] $mmod_port_num] == 1} {
		set rci $pmi
		set t 1
	    }
	    if {$t == 1} {
		break
	    }
	}

	set real_omodid [lindex $rci 0]
	set real_owhich [lindex $rci 1]

	set temp_name "$omodid"
	append temp_name "_p$mmod_port_num"
	
	
	set real_con_name $connid
	set real_con_name [string trimleft $connid $temp_name]

	set new_left_name $real_omodid
	append new_left_name "_p$real_owhich"
	
	append new_left_name "_$real_con_name"
	set real_connid $new_left_name
	
	set templist ""
	
	foreach con $connection_list {
	    set tempid "[lindex $con 0]_p[lindex $con 1]_to_[lindex $con 2]_"
	    append tempid "p[lindex $con 3]"
	    
	    if { [string match $tempid $real_connid] == 0 } {
		set templist "$templist {$con}"
	    }
	    
	}
	
	set connection_list $templist
	
	netedit deleteconnection $real_connid $real_omodid $imodid
	global netedit_canvas netedit_mini_canvas
	$netedit_canvas delete $connid
	$netedit_mini_canvas delete $connid
	configureOPorts $omodid
	configureIPorts $imodid
	
	set templist ""
	foreach fc [$omodid get_FakeConnections] {
	    if { [string match $connid [lindex $fc 0]] == 0 } {
		set templist "$templist {$fc}"
	    }
	}

	$omodid set_FakeConnections $templist
	
    } elseif { [string match [$imodid mod_type] "macromodule"] == 1 } {
	# A little trick to find the macromodule port nmumber
	
	for {set ctr 0} {$ctr <= {eval $ctr+1}} {incr ctr} {
	    set b 0
	    if { [string match "*_to_MacroModule?_p$ctr*" $connid] == 1} {
		set mmod_port_num $ctr
		set b 1
	    }
	    if {$b == 1} {
		break
	    }
	}

	set port_mapping_info [$imodid get_iport_mapping]

	foreach pmi $port_mapping_info {
	    set t 0
	    if { [string match [lindex $pmi 2] $mmod_port_num] == 1 } {
		set rci $pmi
		set t 1
	    }
	    if {$t == 1} {
		break
	    }
	}

	set real_imodid [lindex $rci 0]
	set real_iwhich [lindex $rci 1]

	foreach cinf [netedit getconnected $omodid] {
	    set t 0
	    if { [string match "*$real_imodid*$real_iwhich" [lindex $cinf 0]]\
		    == 1 } {
		set owhich [lindex $cinf 2]
		set t 1
	    }
	    if { $t == 1 } {
		break;
	    }
	}

	set real_connid "$omodid"
	append real_connid "_p$owhich"
	append real_connid "_to_$real_imodid"
	append real_connid "_p$real_iwhich"

	
	set templist ""

		
	global connection_list
	foreach con $connection_list {
	    set tempcon "[lindex $con 0]_p[lindex $con 1]_to_[lindex $con 2]"
	    append tempcon "_p[lindex $con 3]"
	    
	    if { [string match $real_connid $tempcon] == 0 } {
		set templist "$templist {$con}"
	    }
	}

	set connection_list $templist



	netedit deleteconnection $real_connid
	global netedit_canvas netedit_mini_canvas
	$netedit_canvas delete $connid
	$netedit_mini_canvas delete $connid
	
	configureOPorts $omodid
	configureIPorts $imodid
	
	set templist ""
	foreach fc [$imodid get_FakeConnections] {
	    if { [string match $connid [lindex $fc 0]] == 0 } {
		set templist "$templist {$fc}"
	    }
	}

	$imodid set_FakeConnections $templist
		
    } else {
	global connection_list
	set templist ""
	
	foreach con $connection_list {
	    set tempcon "[lindex $con 0]_p[lindex $con 1]_to_[lindex $con 2]"
	    append tempcon "_p[lindex $con 3]"

	    if { [string match $tempcon $connid] == 0 } {
		set templist "$templist {$con}"
	    }
	}

	set connection_list $templist

	

	global netedit_canvas netedit_mini_canvas
	$netedit_canvas delete $connid
	$netedit_mini_canvas delete $connid
	netedit deleteconnection $connid $omodid 
	configureOPorts $omodid
	configureIPorts $imodid
    }
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

proc rebuildConnections {list color} {
    set maincanvas .bot.neteditFrame.canvas
    set minicanvas .top.globalViewFrame.canvas

    global MacroedModules

    set temp_list ""

    if { $MacroedModules != "" } {
	foreach l $list {
	    
	    if { [[lindex $l 1] Macroed] == 1 } {
		rebuildMModuleConnections [[lindex $l 1] MacroModule]
		continue
	    }
	    
	    if { [[lindex $l 3] Macroed] == 1 } {
		rebuildMModuleConnections [[lindex $l 3] MacroModule]
		continue
	    }
	    
	    # if neither module is macroed
	    
	    set temp_list "$temp_list {$l}"
	}
	set list $temp_list
    }   

    foreach i $list {
	$maincanvas raise [lindex $i 0]
	
	if { $color == 1 } {
	    $minicanvas itemconfigure [lindex $i 0] \
		    -fill [$maincanvas itemcget [lindex $i 0] -fill]
	}
	
	set connid [lindex $i 0]
	set omodid [lindex $i 1]
	set owhich [lindex $i 2]
	set imodid [lindex $i 3]
	set iwhich [lindex $i 4]

	rebuildConnection $connid $omodid $owhich $imodid $iwhich
    }
}

proc trackIPortConnection {imodid which x y} {
    global mm
    set mm 0
    if { [string match [$imodid mod_type] "macromodule"] == 1 } {
	set mm 1
	set fake_imodid $imodid
	set fake_iwhich $which
	
	set timodid [lindex [lindex [$imodid get_iport_mapping] $which] 0]
	set tiwhich [lindex [lindex [$imodid get_iport_mapping] $which] 1]
	
	set imodid $timodid
	set which $tiwhich
    }

    # Get coords in canvas
    global netedit_canvas
    set ox1 [winfo x $netedit_canvas.module$imodid.iport$which]
    set ox2 [winfo x $netedit_canvas.module$imodid]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $netedit_canvas.module$imodid.iport$which]
    set oy2 [winfo y $netedit_canvas.module$imodid]
    set y [expr $y+$oy1+$oy2]

    if { $mm == 1 } {
	set c [computeIPortCoords $fake_imodid $fake_iwhich]
    } else {
	set c [computeIPortCoords $imodid $which]
    }

    set ix [lindex $c 0]
    set iy [lindex $c 1]
    set mindist [computeDist $x $y $ix $iy]
    set minport ""
    global new_conn_oports
    


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
    $netedit_canvas itemconfigure tempconnections -fill black

    global potential_connection
    if {$minport != ""} {
	set omodid [lindex $minport 0]
	set owhich [lindex $minport 1]
	$netedit_canvas itemconfigure iconn$owhich$omodid -fill red
	if { $mm == 1 } {
	    set potential_connection [list $omodid $owhich $fake_imodid\
		    $fake_iwhich]
	} else {
	    set potential_connection [list $omodid $owhich $imodid $which]
	}
    } else {
	set potential_connection ""
    }
}

proc trackOPortConnection {omodid which x y} {
    global mm
    set mm 0

    if { [string match [$omodid mod_type] "macromodule"] == 1 } {
	set mm 1
	set fake_omodid $omodid
	set fake_owhich $which
	
	set tomodid [lindex [lindex [$omodid get_oport_mapping] $which] 0]
	set towhich [lindex [lindex [$omodid get_oport_mapping] $which] 1]

	set omodid $tomodid
	set which $towhich
    }

    # Get coords in canvas
    global netedit_canvas
    set ox1 [winfo x $netedit_canvas.module$omodid.oport$which]
    set ox2 [winfo x $netedit_canvas.module$omodid]
    set x [expr $x+$ox1+$ox2]
    set oy1 [winfo y $netedit_canvas.module$omodid.oport$which]
    set oy2 [winfo y $netedit_canvas.module$omodid]
    set y [expr $y+$oy1+$oy2]


    if { $mm == 1 } {
	set c [computeOPortCoords $fake_omodid $fake_owhich]
    } else {
	set c [computeOPortCoords $omodid $which]
    }

    set ix [lindex $c 0]
    set iy [lindex $c 1]


    set relativeMouseX [expr $x+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*4500))])]
    set relativeMouseY [expr $y+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*4500))])]

    set mindist [computeDist $relativeMouseX $relativeMouseY $ix $iy]
    set mindist 6364
    set minport ""
    global new_conn_iports
    
    foreach i $new_conn_iports {
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
	
	if { $mm == 1 } {
	    set potential_connection\
		    [list $fake_omodid $fake_owhich $imodid $iwhich]
	} else {
	    set potential_connection [list $omodid $which $imodid $iwhich]
	}
    } else {
	set potential_connection ""
    }
}

proc endPortConnection {portcolor} {
    global netedit_canvas
    $netedit_canvas delete tempconnections
    $netedit_canvas delete tempname
    destroy $netedit_canvas.frame
    global potential_connection
    if { $potential_connection != "" } {
	if { [string match [[lindex $potential_connection 0] mod_type]\
		"macromodule"] == 1 } {
	    if { [string match [[lindex $potential_connection  2] mod_type]\
		    "macromodule"] == 1 } {
		
		set info1 [get_real_oport [lindex $potential_connection 0]\
			[lindex $potential_connection 1]]
				
		set info2 [get_real_iport [lindex $potential_connection 2]\
			[lindex $potential_connection 3]]
		# The connection oport
		set omodid [lindex $info1 0]
		set owhich [lindex $info1 1]

		# The connection iport
		set imodid [lindex $info2 0]
		set iwhich [lindex $info2 1]

		# Update Connection List
		
		global connection_list
		set connection_list "$connection_list {$omodid $owhich\
			$imodid $iwhich $portcolor}"


		#The MacroModule oport aliases
		set fake_omodid [lindex $potential_connection 0]
		set fake_owhich [lindex $potential_connection 1]

		#The MacroModule iport aliases
		set fake_imodid [lindex $potential_connection 2]
		set fake_iwhich [lindex $potential_connection 3]

		#The aliased connection ID
		set fake_connid $fake_omodid
		append fake_connid "_p$fake_owhich"
		append fake_connid "_to_$fake_imodid"
		append fake_connid "_p$fake_iwhich"
		
		# Build the "fake" connection
		buildConnection $fake_connid $portcolor $fake_omodid\
			$fake_owhich $fake_imodid $fake_iwhich
		
		# Append the "fake" connections to their respective
		# MacroModule's lists of fake connections

		$fake_omodid set_FakeConnections\
			"[$fake_omodid get_FakeConnections]\
			{$fake_connid $fake_omodid $fake_owhich\
			$fake_imodid $fake_iwhich}"
		
		$fake_imodid set_FakeConnections\
			"[$fake_imodid get_FakeConnections]\
			{$fake_connid $fake_omodid $fake_owhich\
			$fake_imodid $fake_iwhich}"
		

		netedit addconnection $omodid $owhich $imodid $iwhich
		
		# Append connection information to a list of connections
		# to be rebuilt when connections are destroyed

		configureOPorts $fake_omodid
		configureIPorts $fake_imodid
	    		
	    }  else {
		# The output module [only] is of type macro
		set info [get_real_oport [lindex $potential_connection 0]\
			[lindex $potential_connection 1]]

		# The connection oport
		set omodid [lindex $info 0]
		set owhich [lindex $info 1]

		# The connection iport
		set imodid [lindex $potential_connection 2]
		set iwhich [lindex $potential_connection 3]
		
		# Update connection_list
		global connection_list
		set connection_list "$connection_list {$omodid $owhich $imodid\
			$iwhich $portcolor}"

		
		# The macromodule oport aliases
		set fake_omodid [lindex $potential_connection 0]
		set fake_owhich [lindex $potential_connection 1]

		# The aliased connection ID
		set fake_connid \
			"[lindex $potential_connection 0]_p[lindex $potential_connection 1]_to_[lindex $potential_connection 2]_p[lindex $potential_connection 3]"
		
		# Build the "fake" connection
		buildConnection $fake_connid $portcolor $fake_omodid\
			$fake_owhich $imodid $iwhich
		
		# Append the "fake" connection to its respective macromodule's
		# list of fake connections
		$fake_omodid set_FakeConnections\
			"[$fake_omodid get_FakeConnections]\
			{$fake_connid $fake_omodid $fake_owhich\
			$imodid $iwhich}"
		
		netedit addconnection $omodid $owhich $imodid $iwhich
		
		configureOPorts $fake_omodid
		configureIPorts $imodid

	    }
	} elseif { [string match [[lindex $potential_connection 2] mod_type]\
		"macromodule"] == 1 } {
	    	    
	    # The input module [only] is of type macro
	    set info [get_real_iport [lindex $potential_connection 2]\
		    [lindex $potential_connection 3]]

	    
	    # The connection oport
	    set omodid [lindex $potential_connection 0]
	    set owhich [lindex $potential_connection 1]

	    # The connection iport
	    set imodid [lindex $info 0]
	    set iwhich [lindex $info 1]

	    # Update connection_list
	    global connection_list
	    set connection_list "$connection_list {$omodid $owhich $imodid\
		    $iwhich $portcolor}"


	    # The macromodule iport aliases
	    set fake_imodid [lindex $potential_connection 2]
	    set fake_iwhich [lindex $potential_connection 3]


	    # The aliased connection ID
	    set fake_connid\
		    "[lindex $potential_connection 0]_p[lindex $potential_connection 1]_to_[lindex $potential_connection 2]_p[lindex $potential_connection 3]"
	    # Build the fake connection
	    buildConnection $fake_connid $portcolor $omodid $owhich\
		    [lindex $potential_connection 2]\
		    [lindex $potential_connection 3]
	    # As no widget exists on the canvas for the real module
	    # at this time, place it in a list of connection icons
	    # to be build when either the macromodule is expanded, or
	    # the canvas is saved
	    
	    $fake_imodid set_FakeConnections\
		    "[$fake_imodid get_FakeConnections]\
		    {$fake_connid $omodid $owhich\
		    $fake_imodid $fake_iwhich}"


	    
	    global MModuleFakeConnections
	    
	    netedit addconnection $omodid $owhich $imodid $iwhich
	    
	    configureOPorts $omodid
	    configureIPorts $fake_imodid
	} else {
	    # Normal connection; no macromods
	    set omodid [lindex $potential_connection 0]
	    set owhich [lindex $potential_connection 1]
	    set imodid [lindex $potential_connection 2]
	    set iwhich [lindex $potential_connection 3]

	    # Update connection_list
	    global connection_list
	    set connection_list "$connection_list {$omodid $owhich $imodid\
		    $iwhich $portcolor}"
	    

	    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
	    buildConnection $connid $portcolor $omodid $owhich $imodid $iwhich
	    
	    configureOPorts $omodid
	    configureIPorts $imodid
	}
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
    
    # For compatibility with macromodules
    
    if {[string match "*MacroModule*" $modid]} {
	set h [winfo height $netedit_canvas.macromodule$modid]
    } else {
	set h [winfo height $netedit_canvas.module$modid]
    }
    
    set x [expr $px+$mx]
    set y [expr $my+$h]
    return [list $x $y]
}

proc computeDist {x1 y1 x2 y2} {
    set dx [expr $x2-$x1]
    set dy [expr $y2-$y1]
    return [expr sqrt($dx*$dx+$dy*$dy)]
}

proc generate_current_connections {} {
    global connection_list
    
    set c_list ""
    
    foreach c $connection_list {
	set color [lindex $c 4] 
	set oConinf "[[lindex $c 0] get_oports [lindex $c 1]]"
	set oConinf [string trim $oConinf "{}"]
	set omodid [lindex $oConinf 0]
	set owhich [lindex $oConinf 1]
	if { [string match $omodid ""] } {
	    continue
	}
	

	set iConinf [[lindex $c 2] get_iports [lindex $c 3]]
	set iConinf [string trim $iConinf "{}"]
	set imodid [lindex $iConinf 0]
	set iwhich [lindex $iConinf 1]
	if { [string match $imodid ""] } {
	    continue
	}
	
	set connid $omodid
	append connid "_p$owhich"
	append connid "_to_$imodid"
	append connid "_p$iwhich"
	
	set c_list "$c_list {$connid $color $omodid $owhich $imodid $iwhich}"
    }


    return $c_list

}





proc moduleStartDrag {maincanvas modid x y} {
    global lastX lastY
    global module_group
    global CurrentMacroModules
    global CurrentlySelectedModules
    global current_connections
    global sel_module_box
    
    set wname "$maincanvas.module$modid"
    raise $wname

    set sel_module_box 0

    set module_group [string match "*$modid*" $CurrentlySelectedModules]
    set lastX $x
    set lastY $y
    

    

    if {$module_group == 1} {
	foreach csm $CurrentlySelectedModules {
	    $csm set_moduleDragged 0
	    $csm set_moduleConnected [netedit getconnected $modid]
	}
    } else {
	$modid set_moduleDragged 0
	$modid set_moduleConnected [netedit getconnected $modid]
    }  
    
    set current_connections ""
    
    set count 0
    
    foreach csm $CurrentlySelectedModules {
	incr count
	foreach i [netedit getconnected $csm] {
	    set l [lindex $i 0]
	    if {[string match "*$l*" $current_connections] == 0} {

	    }
	}
}       


  # Hide the network lines which are not to be redrawn during
  # the drag
  if { ($count >= 2)&&([string match "*$modid*" $CurrentlySelectedModules]==1) } {
      #Draw Box
      set bbox [compute_bbox $maincanvas]
      set sel_module_box [$maincanvas create rectangle [lindex $bbox 0] \
	      [lindex $bbox 1] [lindex $bbox 2] [lindex $bbox 3]]
      $maincanvas itemconfigure $sel_module_box -outline darkgray
      
      foreach c $current_connections {
	  $maincanvas lower $c
	  .top.globalViewFrame.canvas itemconfigure $c -fill #036
      }
  }
}


proc moduleDrag {maincanvas minicanvas modid x y} {
    global module_group
    global grouplastX
    global grouplastY
    global lastX
    global lastY
    global CurrentlySelectedModules
    global sel_module_box

    
    if {$module_group == 1 } {
	
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


	####################
	
	foreach csm $CurrentlySelectedModules {
	    do_moduleDrag $maincanvas $minicanvas $csm $x $y
	}	
	
	set lastX $grouplastX
	set lastY $grouplastY
		
    } else {
	do_moduleDrag $maincanvas $minicanvas $modid $x $y
	set lastX $grouplastX
	set lastY $grouplastY
    }
}    


proc do_moduleDrag {maincanvas minicanvas modid x y} {
    global module_group
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
    
   
    #Rebuild the network connections only if one module is selected

    set i 0

    if {$module_group == 1} {
	foreach csm $CurrentlySelectedModules {
	    $csm set_moduleDragged 1
	    incr i
	}
	if {$i <= 1} {
	    rebuildConnections [$csm get_moduleConnected] 0
	}
    } else {
	$modid set_moduleDragged 1
	rebuildConnections [$modid get_moduleConnected] 0
    }
    
    set i 0
    
    set bbox [compute_bbox $maincanvas]

    if { $sel_module_box != 0 } {
	    $maincanvas coords $sel_module_box [lindex $bbox 0] \
		    [lindex $bbox 1] [lindex $bbox 2] [lindex $bbox 3]
    }

}

proc moduleEndDrag {mframe maincanvas} {
    global module_group
    global CurrentlySelectedModules
    global current_connections
    global sel_module_box





    #Update all network lines on the canvas if a group of modules was moved
    

    global MacroedModules

    if {$module_group == 1} {
	set maincanvas .bot.neteditFrame.canvas
	global mainCanvasWidth mainCanvasHeight
	
	set overlap [$maincanvas find overlapping 0 0 $mainCanvasWidth \
		$mainCanvasHeight] 
	foreach i  $overlap {
	    set s "[$maincanvas itemcget $i -tags]"
	    if {[$maincanvas type $s] == "window"} {
		if {[string match [$s mod_type] "module"]} {
		    rebuildConnections [netedit getconnected $s] 0
		}
	    }
	}		
    }

    

    


    # Raise the lines hidden during the drag; do not
    # raise the lines hidded because of a MacroModule
    foreach c $current_connections {
	set t 0
	foreach m $MacroedModules {
	    if { [string match "*$m*" $c] == 1 } {
		set t 1
	    }
	}
	if {$t == 0} {
	    $maincanvas raise $c
	}
    }



    foreach c $current_connections {
 	$maincanvas raise $c
	.top.globalViewFrame.canvas itemconfigure $c -fill \
		[$maincanvas itemcget $c -fill]
    }

    $maincanvas delete $sel_module_box

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
    
    # Remove me from the modules list

    global modules
    set templist ""
    foreach m $modules {
	if { ! [string match $m $modid] } {
	    set templist "$templist $m"
	}
    }

    set modules $templist

    #Remove me from the Currently Selected Module List
	global CurrentlySelectedModules
	set p $maincanvas.module[$modid modname].ff
	
	set tempList ""
    
	foreach item $CurrentlySelectedModules {
	    if { $item != [$modid modname] } {
		set tempList "$tempList $item"
	    }
	}
	
	set CurrentlySelectedModules $tempList
    
    set modList [netedit getconnected $modid]
    foreach i $modList {
	set connid [lindex $i 0]
	set omodid [lindex $i 1]
	set owhich [lindex $i 2]
	set imodid [lindex $i 3]
	set iwhich [lindex $i 4]
	set iinfo [lindex [lindex [$omodid get_oports $owhich] 0] 0]

	set romodid [lindex [lindex [$omodid get_oports $owhich] 0] 0]
	set rowhich [lindex [lindex [$omodid get_oports $owhich] 0] 1]
	set rimodid [lindex [lindex [$imodid get_iports $iwhich] 0] 0]
	set riwhich [lindex [lindex [$imodid get_iports $iwhich] 0] 1]
	
	set rconnid "$romodid"
	append rconnid "_p$rowhich"
	append rconnid "_to_$rimodid"
	append rconnid "_p$riwhich"
	
	destroyConnection $rconnid $romodid $rimodid

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

proc moduleDestroySelected {maincanvas minicanvas module} {
    global CurrentlySelectedModules 
       
    foreach mnum $CurrentlySelectedModules {
	moduleDestroy $maincanvas $minicanvas $mnum
    }
}

global startx starty
global rect
global rectlastx rectlasty

global TempList

proc startBox {X Y maincanvas option} {
    global CurrentlySelectedModules
    global mainCanvasWidth mainCanvasHeight
    global startx starty
    global TempList
    global rect
    global rectlastx rectlasty
    global selected_color unselected_color
    
    set TempList ""
    

    if { $option == 0 } {
	foreach i $CurrentlySelectedModules {
	    $i removeSelected $maincanvas $unselected_color
	}
	set $CurrentlySelectedModules ""
    } 

    
    set sx [expr $X-[winfo rootx $maincanvas]]
    set sy [expr $Y-[winfo rooty $maincanvas]]
    
    set startx [expr $sx+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*$mainCanvasWidth))])]
    
    set starty [expr $sy+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
  
    #Begin the bounding box
    set rect [$maincanvas create rectangle $startx $starty $startx $starty] 
    
    set rectlastx $X
    set rectlasty $Y
    
    global rx ry

    set rx [winfo rootx $maincanvas]
    set ry [winfo rooty $maincanvas]
    
}

proc makeBox {X Y maincanvas} {
    global CurrentlySelectedModules
    global mainCanvasWidth mainCanvasHeight
    global selected_color unselected_color
    global TempList
    global startx
    global starty 
    global rect
    global rx ry
    
    #Canvas Relative current X and Y positions

    set tx [expr ($X-$rx)]
    set ty [expr ($Y-$ry)]
    
    set currx [expr $tx+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*$mainCanvasWidth))])]
    
    set curry [expr $ty+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
   
    $maincanvas coords $rect $startx $starty $currx $curry
    

    # select all modules which overlap the current bounding box
    
    set ol ""

    set overlap [$maincanvas find overlapping $startx $starty $currx $curry]
    foreach i $overlap {
	set s "[$maincanvas itemcget $i -tags]"
	if {[$maincanvas type $s] == "window"} {
	    if { [string match [$s mod_type] "module"] } {
		$s addSelected $maincanvas $selected_color
		set TempList "$TempList $s"    
		set ol "$ol $s"
	    }
	}
    }


    #For each element of $TempList, check to see if it is still selected.  If
    #so, leave it selected and in TempList.  If not, remove if from the
    #templist, and unselect it.
    
    set ttlist ""
    

    foreach t $TempList {
	if { [string match "*$t*" $ol] == 1 } {
	    set ttlist "$ttlist $t"
	} else {
	    $t removeSelected $maincanvas $unselected_color
	}
    }
    
    #Update TempList
    set TempList $ttlist


}

proc endBox {X Y maincanvas} {
    global rect
    global TempList
    $maincanvas delete $rect
    set TempList ""
}

proc SelectAll {} {
    set maincanvas .bot.neteditFrame.canvas
    global mainCanvasWidth mainCanvasHeight
    global selected_color

    set overlap [$maincanvas find overlapping 0 0 $mainCanvasWidth \
	    $mainCanvasHeight] 
    foreach i $overlap {
	set s "[$maincanvas itemcget $i -tags]"
	
	if {[$maincanvas type $s] == "window"} {
	    $s addSelected $maincanvas $selected_color
	}
    }
}
    
proc ClearCanvas {} {
    set maincanvas .bot.neteditFrame.canvas
    set minicanvas .top.globalViewFrame.canvas
    global mainCanvasWidth mainCanvasHeight
    
    set overlap [$maincanvas find overlapping 0 0 $mainCanvasWidth \
	    $mainCanvasHeight] 
    foreach i $overlap {
	set s "[$maincanvas itemcget $i -tags]"
	 
	if {[$maincanvas type $s] == "window"} {
	    moduleDestroy $maincanvas $minicanvas $s
	}
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



############

proc makeMacroModule {maincanvas minicanvas curr_module} {
    set mmod [groupModules $maincanvas $minicanvas $curr_module]
}

proc groupModules {maincanvas minicanvas curr_module} {
    createMacroModule $maincanvas $minicanvas $curr_module 1 0 0
}

proc createMacroModule {maincanvas minicanvas curr_module at_cursor xpos ypos } {
    global CurrentlySelectedModules
    
    
    # Execute this procedure only if there are selected modules
    set i 0
    foreach csm $CurrentlySelectedModules {
	incr i
    }

    if { $i > 0 } {

	# Calculate the coordinates of the current module, and use them as the 
	# position at which to create the new MacroModule
	if { $at_cursor == 1 } {
	    set xpos [$curr_module get_x]
	    set ypos [$curr_module get_y]
	}
	
	set m MacroModule[mmodnum]
	
	macromodule $m
	$m set_members $CurrentlySelectedModules
	
    
	####
	
	set mmodule_connections ""

	foreach csm $CurrentlySelectedModules {
	    foreach i [netedit getconnected $csm] {
		set l [lindex $i 0]
		if {[string match "*$l*" $mmodule_connections] == 0} {
		    set mmodule_connections "$mmodule_connections $l"
		}
	    }
	}   
    
	# Hide the "real" network lines
	foreach mod $mmodule_connections {
	    $maincanvas delete $mod
	    $minicanvas delete $mod
	    
	    #$maincanvas lower $mod
	    #$minicanvas itemconfigure $mod -fill #036
	}

	#$m set_connections $mmodule_connections
    
	
	# Tag the selected Module classes with the name of the MacroModule
	# they are joining
	
	foreach csm $CurrentlySelectedModules {
	    $csm set_Macroed 1
	    $csm set_MacroModule $m
	}

	# Hide the "real" modules
	# At this point, there is no way to "unmap" window objects,
	# so, as a temporary fix, modules are uniformly moved up and to
	# the right by 10^3 pixels
	
	foreach csm $CurrentlySelectedModules {
	    
	    $csm set_last_pos "[$csm get_x] [$csm get_y]"
	    $maincanvas delete $csm
	    $minicanvas itemconfigure $csm -fill #036
	}
	
	# Unselect the Modules which comprise the newly created MacroModule
	global unselected_color

	foreach i $CurrentlySelectedModules {
	    $i removeSelected $maincanvas $unselected_color
	}
	set $CurrentlySelectedModules ""

	# Make and pack the new macromodule's icon
	$m make_icon $maincanvas $minicanvas $xpos $ypos
	
    }
    
    rebuildMModuleConnections $m
    
    return $m
}

# Assign an ID number to a macromodule
proc mmodnum  {} {
    global CurrentMacroModules
    set num "1"
    while {1} {
	if {[string match "*$num*" $CurrentMacroModules] == 1} {
	    incr num
	} else {
	    return $num
	}
    }
}

proc ungroup_modules {maincanvas minicanvas mmodid} {
    global MacroedModules
    global CurrentMacroModules
    global SCALEX SCALEY
    global mainCanvasWidth mainCanvasHeight

    set fcons [$mmodid get_FakeConnections]

    # Destroy the macromodule's icon
    $maincanvas delete $mmodid
    $minicanvas delete $mmodid

    set mems [$mmodid get_members]

    # If the selected module is connected to another
    # macromodule, modify that module's connections
    # appropriately

    foreach c [$mmodid get_FakeConnections] {
	if { [string match [lindex $c 1] $mmodid] } {
	    if { [string match [[lindex $c 3] mod_type] "macromodule"] } {
		foreach mfc [[lindex $c 3] get_FakeConnections] {
		    if { [string match [lindex $mfc 0] [lindex $c 0]] } {
			set old_con_name [lindex $mfc 0]
			set omodid [lindex $mfc 1]
			set owhich [lindex $mfc 2]
			set imodid [lindex $mfc 3]
			set iwhich [lindex $mfc 4]
			foreach inf [$mmodid get_oport_mapping] {
			    if { [string match [lindex $inf 2] $owhich] } {
				set real_omodid [lindex $inf 0]
				set real_owhich [lindex $inf 1]
			    }
			}
			set con_name $real_omodid
			append con_name "_p$real_owhich"
			append con_name "_to_$imodid"
			append con_name "_p$iwhich"
			
			set templist ""
			
			foreach nmfc [$imodid get_FakeConnections] {
			    if { [string match $old_con_name [lindex $nmfc 0]\
				    ] } {
				set templist "$templist {$con_name\
					$real_omodid $real_owhich $imodid\
					$iwhich}"
			    } else {
				set templist "$templist {$nmfc}"
			    }
			}
			
			$imodid set_FakeConnections $templist
		    }
		}
	    }
	}
    

	if { [string match [lindex $c 3] $mmodid] } {
	    if { [string match [[lindex $c 1] mod_type] "macromodule"] } {
		foreach mfc [[lindex $c 1] get_FakeConnections] {
		    if { [string match [lindex $mfc 0] [lindex $c 0]] } {
			set old_con_name [lindex $mfc 0]
			set omodid [lindex $mfc 1]
			set owhich [lindex $mfc 2]
			set imodid [lindex $mfc 3]
			set iwhich [lindex $mfc 4]
			foreach inf [$mmodid get_iport_mapping] {
			    if { [string match [lindex $inf 2] $iwhich] } {
				set real_imodid [lindex $inf 0]
				set real_iwhich [lindex $inf 1]
			    }
			}

			set con_name $omodid
			append con_name "_p$owhich"
			append con_name "_to_$real_imodid"
			append con_name "_p$real_iwhich"
			
			set templist ""

			foreach nmfc [$omodid get_FakeConnections] {
			    if { [string match $old_con_name [lindex $nmfc 0]\
				    ] } {
				set templist "$templist {$con_name\
					$omodid $owhich $real_imodid\
					$real_iwhich}"
			    } else {
				set templist "$templist {$nmfc}"
			    }
			
			    $omodid set_FakeConnections $templist
			}
		    }
		}
	    }
	}
    }

    # Untag the modules
    foreach member $mems {
	$member set_Macroed 0
	$member set_MacroModule ""
	

	foreach c [$mmodid get_FakeConnections] {
	    $maincanvas delete [lindex $c 0]
	    $minicanvas delete [lindex $c 0]
	}
	
	$mmodid set_FakeConnections ""
    }

    foreach member $mems {
	set last_x [lindex [$member get_last_pos] 0]
	set last_y [lindex [$member get_last_pos] 1]
	
	
	# Place the modules back on the canvas in their
	# Original Position(s)

	set xv [lindex [$maincanvas xview] 0]
	set yv [lindex [$maincanvas yview] 0]
		
	set last_x [expr $last_x+($xv*$mainCanvasWidth)]
	set last_y [expr $last_y+($yv*$mainCanvasHeight)]

	$maincanvas create window $last_x $last_y -window \
		$maincanvas.module$member \
		-tags $member -anchor nw    

	# $minicanvas itemconfigure $member -fill gray
	
	$minicanvas create rectangle \
		[expr $last_x/$SCALEX] [expr $last_y/$SCALEY] \
		[expr $last_x/$SCALEX+4] [expr $last_y/$SCALEY + 2]\
		-outline "" -fill gray -tags $member
	





	# Remove member from MacroedModules list
	set tlist ""
	foreach m $MacroedModules {
	    
	    if { [string match $member $m] == 0 } {
		set tlist "$tlist $m"
	    }
	    set MacroedModules $tlist
	}

	#configureOPorts $member
	#configureIPorts $member
    }

    destroy ${maincanvas}.macromodule$mmodid
    destroy ${minicanvas}.macromodule$mmodid
    

    
    # Rebuild the connections
    foreach curr_con [generate_current_connections] {
	if { [string match [$maincanvas gettags [lindex $curr_con 0]]\
		[lindex $curr_con 0]] == 0 } {

	    buildConnection [lindex $curr_con 0] [lindex $curr_con 1]\
		    [lindex $curr_con 2] [lindex $curr_con 3]\
		    [lindex $curr_con 4] [lindex $curr_con 5]	    

	    configureOPorts [lindex $curr_con 2]
	    configureIPorts [lindex $curr_con 4]
	} 
    }
    

    foreach member [$mmodid get_members] {
	configureOPorts $member
	configureIPorts $member
    }


    $mmodid delete
    

    global CurrentMacroModules
    
    set templist ""
    foreach cmm $CurrentMacroModules {
	if { [string match $mmodid $cmm] == 0 } {
	    set templist "$templist $cmm"
	}
    }
    
    set CurrentMacroModules $templist
    
}



proc get_real_iport { mmodid iwhich } {
    set mapping [$mmodid get_iport_mapping]
    
    foreach m $mapping {
	if { [string match [lindex $m 2] $iwhich] == 1 } {
	    set info "[lindex $m 0] [lindex $m 1]"
	    return $info
	}
    }

    # we should never get this far...
}

proc get_real_oport { mmodid owhich } {
    set mapping [$mmodid get_oport_mapping]
    
    foreach m $mapping {
	if { [string match [lindex $m 2] $owhich] == 1 } {
	    set info "[lindex $m 0] [lindex $m 1]"
	    return $info
	}
    }
    
    # we should never get this far...
}

