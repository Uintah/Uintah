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
#  Subnet.tcl
#
#  Written by:
#   McKay Davis
#   Department of Computer Science
#   University of Utah
#   September 2003
#
#  Copyright (C) 2003 SCI Group
#


itcl_class SubnetModule {
    inherit Module
    constructor { config } {
	set make_progress_graph 0
	set make_time 0
	set isSubnetModule 1 
    }

    destructor {
	destroy .subnet$subnetNumber
    }
	
    method ui {} {
	global Subnet
	foreach modid $Subnet(Subnet${subnetNumber}_Modules) {
	    if { [$modid have_ui] } {
		$modid initialize_ui
	    }
	}
    }

    method execute {} {
	global Subnet
	foreach modid $Subnet(Subnet${subnetNumber}_Modules) {
	    $modid execute
	}
    }

    method update_msg_state {} { 
	global Subnet 
	set state 0
	set msg_open ""
	foreach modid $Subnet(Subnet${subnetNumber}_Modules) {
	    switch [$modid get_msg_state] {
		Error   { lappend msg_open $modid; set state 3 }
		Warning { lappend msg_open $modid; if {$state<2} {set state 2}}
		Remark  { if {$state < 1} {set state 1}}
		Reset   { }
		default { }
	    }
	}

	switch $state {
	    3 { set color red }
	    2 { set color yellow }
	    1 { set color blue }
	    0 { set color grey75 }
	}

	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set indicator $canvas.module[modname].ff.msg.indicator
	place forget $indicator
	$indicator configure -width $indicator_width -background $color
	place $indicator -relheight 1 -anchor nw 
	bind $indicator <Button> "foreach modid {$msg_open} {\$modid displayLog}"
	
	
	if {[winfo exists .standalone]} {
	    app indicate_error [modname] $msg_state
	}
	
    }


	
}
       
proc updateSubnetName { subnet name1 name2 op } {
    global Subnet
    # extract the Subnet Number from the array index
    # set subnet [string range $name2 6 [expr [string first _Name $name2]-1]]
    set w .subnet${subnet}
    if [winfo exists $w] {
	wm title $w "$Subnet($name2) Sub-Network Editor"
    }
    if { [llength [info command SubnetIcon${subnet}]] } { 
	SubnetIcon${subnet} setColorAndTitle
    }
}


proc makeSubnet { from_subnet x y { bbox "0 0 0 0" }} {
    global Subnet mainCanvasHeight mainCanvasWidth Color

    # Setup default Subnet Variables
    incr Subnet(num)
    set Subnet(Subnet$Subnet(num))		$Subnet(num)
    set Subnet(Subnet$Subnet(num)_Name)		"Sub-Network \#$Subnet(num)"
    set Subnet(Subnet$Subnet(num)_Modules)	""
    set Subnet(Subnet$Subnet(num)_connections)	""

    # Automatically update icon and window title when Subnet name changes
    trace variable Subnet(Subnet$Subnet(num)_Name) w \
	"updateSubnetName $Subnet(num)"

    set w .subnet$Subnet(num)
    
    # TODO: probably should raise Subnet window
    if {[winfo exists $w]} { puts here; wm deiconfiy $w; return }

    # Create the Subnet window and hide it
    toplevel $w -width [getAdjWidth $bbox] -height [getAdjHeight $bbox] 
    wm withdraw $w

    wm title $w "$Subnet(Subnet$Subnet(num)_Name) Sub-Network Editor"
    wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"
    update idletasks

    # Make the Subnet Menu Bar
    frame $w.main_menu -relief raised -borderwidth 3

    # Make the File menu item
    menubutton $w.main_menu.file -text "File" -underline 0 \
	-menu $w.main_menu.file.menu
    menu $w.main_menu.file.menu -tearoff false
    $w.main_menu.file.menu add command -label "Save As Template..." \
	-underline 0 -command "saveSubnet $Subnet(num)"
    pack $w.main_menu.file -side left

    # Make the Edit menu item
    menubutton $w.main_menu.edit -text "Edit" -underline 0 \
	-menu $w.main_menu.file.menu
#    menu $w.main_menu.file.edit -tearoff false
    pack $w.main_menu.edit -side left


    # Make the Packages menu item
    menubutton $w.main_menu.packages -text "Packages" -underline 0 \
	-menu $w.main_menu.file.packages
    menu $w.main_menu.file.packagesnew -tearoff false
    pack $w.main_menu.packages -side left


    # Make the Subnet Name Entry Field
    frame $w.fname -borderwidth 2
    label $w.fname.label -text "Name"
    entry $w.fname.entry -validate all -textvariable Subnet(Subnet$Subnet(num)_Name)
    pack $w.fname.label $w.fname.entry -side left -pady 5

    # Make the Subnet Canvas
    frame $w.can -relief flat -borderwidth 0
    frame $w.can.can -relief sunken -borderwidth 3
    pack $w.can.can -fill both -expand yes -pady 8
    set Subnet(Subnet$Subnet(num)_canvas) "$w.can.can.canvas"
    set canvas $Subnet(Subnet$Subnet(num)_canvas)
    canvas $canvas -bg $Color(SubnetEditor) \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight"
    pack $canvas -expand yes -fill both

    # Create a BOGUS minicanvas for Subnet/Main Editor code compatibility
    set Subnet(Subnet$Subnet(num)_minicanvas) $w.can.minicanvas
    canvas $Subnet(Subnet$Subnet(num)_minicanvas)
    
    # Make the background square in the canvas to catch all mouse events
    $canvas create rectangle 0 0 $mainCanvasWidth $mainCanvasWidth \
	-fill $Color(SubnetEditor) -tags "bgRect"

    # Create the Canvas Scrollbars
    scrollbar $w.hscroll -relief sunken -orient horizontal \
	-command "$canvas xview"
    scrollbar $w.vscroll -relief sunken -command "$canvas yview" 

    # Configure the Subnet Editor Canvas Scrollbars
    $canvas configure \
	-yscrollcommand "drawSubnetConnections $Subnet(num);$w.vscroll set" \
	-xscrollcommand "drawSubnetConnections $Subnet(num);$w.hscroll set"

    set Subnet(Subnet$Subnet(num)_state) embedded
    frame $w.state
    radiobutton $w.state.embedded -text Embedded -variable Subnet(Subnet$Subnet(num)_state) \
	-value embedded 
    radiobutton $w.state.ondisk -text "On Disk" -variable Subnet(Subnet$Subnet(num)_state) \
	-value ondisk
    pack $w.state.embedded $w.state.ondisk -side left -fill x -expand yes

    # Create Grid Layout for all Items in Subnet Editor Window
    grid $w.can $w.hscroll $w.vscroll $w.fname	
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure    $w 0 -weight 0 
    grid rowconfigure    $w 0 -weight 0
    grid rowconfigure    $w 2 -weight 1
    grid rowconfigure    $w 3 -weight 0
    grid rowconfigure    $w 4 -weight 0
    grid config $w.main_menu -column 0 -row 0 \
	    -columnspan 2 -rowspan 1 -sticky nwe
    grid config $w.fname -column 0 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky news
    grid config $w.can -column 0 -row 2 \
	    -columnspan 1 -rowspan 1 -sticky news 
    grid config $w.hscroll -column 0 -row 3 \
	    -columnspan 1 -rowspan 1 -sticky ew -pady 2
    grid config $w.vscroll -column 1 -row 2 \
	    -columnspan 1 -rowspan 1 -sticky sn -padx 2
    grid config $w.state -column 0 -row 4\
	    -columnspan 2 -rowspan 1 -sticky sn -padx 2


    # Create the icon for the new Subnet on the old canvas
    set Subnet(SubnetIcon$Subnet(num)) $from_subnet
    SubnetModule SubnetIcon$Subnet(num) \
	-name $Subnet(Subnet$Subnet(num)_Name) -subnetNumber $Subnet(num)
    set Subnet(SubnetIcon$Subnet(num)_num) $Subnet(num)
    set Subnet(SubnetIcon$Subnet(num)_connections) ""
    SubnetIcon$Subnet(num) make_icon $x $y 1
    lappend Subnet(Subnet${from_subnet}_Modules) SubnetIcon$Subnet(num)

    # Select the item in focus, and unselect all others
    $canvas bind bgRect <3> "modulesMenu $Subnet(num) %x %y"
    $canvas bind bgRect <1> "startBox $canvas %X %Y 0"
    $canvas bind bgRect <Control-Button-1> "startBox $canvas %X %Y 1"
    $canvas bind bgRect <B1-Motion> "makeBox $canvas %X %Y"
    $canvas bind bgRect <ButtonRelease-1> "$canvas delete tempbox"

    # SubCanvas up-down bound to mouse scroll wheel
    bind $w <ButtonPress-5>  "canvasScroll $canvas 0.0 0.01"
    bind $w <ButtonPress-4>  "canvasScroll $canvas 0.0 -0.01"

    # Canvas movement on arrow keys press
    bind $w <KeyPress-Down>  "canvasScroll $canvas 0.0 0.01"
    bind $w <KeyPress-Up>    "canvasScroll $canvas 0.0 -0.01"
    bind $w <KeyPress-Left>  "canvasScroll $canvas -0.01 0.0"
    bind $w <KeyPress-Right> "canvasScroll $canvas 0.01 0.0" 
    bind $canvas <Configure> "drawSubnetConnections $Subnet(num)"

    bind all <Control-d> "moduleDestroySelected"
    bind all <Control-l> "ClearCanvas 1 $Subnet(num)"
    bind all <Control-z> "undo"
    bind all <Control-a> "selectAll $Subnet(num)"
    bind all <Control-y> "redo"
    return $Subnet(num)
}



proc createSubnet { from_subnet { modules "" } } {
    global Subnet CurrentlySelectedModules mouseX mouseY Notes backupNotes
    array unset backupNotes
    if { [llength $CurrentlySelectedModules ] } {
	set modules $CurrentlySelectedModules
    }

    # First, delete the connections and module icons from the old canvas
    set connections ""
    foreach modid $modules {
	listFindAndRemove Subnet(Subnet${from_subnet}_Modules) $modid
	eval lappend connections $Subnet(${modid}_connections)
    }

    # remove all duplicate connections from the list
    set connections [lsort -unique $connections]
    # delete connections in decreasing input port number for dynamic ports
    foreach conn [lsort -decreasing -integer -index 3 $connections] {
	array set backupNotes [array get Notes [makeConnID $conn]*]
	destroyConnection $conn 0 0
    }    
    
    # only the first two coordinates matter
    set bbox "$mouseX $mouseY [expr $mouseX+100] [expr $mouseY+100]"
    if { $modules != "" } {
	set canvas $Subnet(Subnet${from_subnet}_canvas)
	set bbox [compute_bbox $canvas $modules]
    }

    set subnet [makeSubnet $from_subnet [lindex $bbox 0] [lindex $bbox 1] $bbox]
    set Subnet(Subnet${subnet}_Modules) $modules
    # Then move the modules to the new canvas
    foreach modid $modules {
	set canvas $Subnet(Subnet$Subnet($modid)_canvas)
	set modbbox [$canvas bbox $modid]
	$canvas delete $modid $modid-notes $modid-notes-shadow
	destroy $canvas.module$modid
	set x [expr [lindex $modbbox 0] - [lindex $bbox 0] + 10]
	set y [expr [lindex $modbbox 1] - [lindex $bbox 1] + 25]
	set Subnet($modid) $subnet
	$modid make_icon $x $y 1
    }

    # Create new connections to Subnet Icon and within Subnet Editor
    # Note the 0 0 parameters to createConnection:  SCIRun wont be
    # notified of any creation or deletion of connections between modules
    # since no actual connections are being changed
    set connections [sortPorts $subnet $connections]
    while { [llength $connections] } {
	# pop the connection off the end of the list
	set conn [lindex $connections 0]
	set connections [lrange $connections 1 end]
	
	if { $Subnet([oMod conn]) == $Subnet([iMod conn]) } {
	    createConnection $conn 0 0
	    array set Notes [array get backupNotes [makeConnID $conn]]
	    continue
	}

	if { $Subnet([oMod conn]) == ${subnet} } {
	    # Split the connection across the subnet boundary
	    set which [portCount "Subnet${subnet} 0 i"]
	    set newconn "[oMod conn] [oNum conn] Subnet${subnet} $which"
	    createConnection $newconn 0 0	    
	    set newconn "SubnetIcon${subnet} $which [iMod conn] [iNum conn]"
	    createConnection $newconn 0 0
	    # take care of connection notes
	    set oldID [makeConnID $conn]
	    set newID [makeConnID $newconn]
	    setIfExists Notes($newID) backupNotes($oldID)
	    setIfExists Notes($newID-Position) backupNotes($oldID-Position)
	    setIfExists Notes($newID-Color) backupNotes($oldID-Color)
	    drawNotes $newID
	} elseif { $Subnet([iMod conn]) == ${subnet} } {
	    set which [portCount "Subnet${subnet} 0 o"]
	    set newconn "[oMod conn] [oNum conn] SubnetIcon${subnet} $which"
	    createConnection $newconn 0 0
	    set newconn "Subnet${subnet} $which [iMod conn] [iNum conn]"
	    createConnection $newconn 0 0
	    # take care of connection notes
	    set oldID [makeConnID $conn]
	    set newID [makeConnID $newconn]
	    setIfExists Notes($newID) backupNotes($oldID)
	    setIfExists Notes($newID-Position) backupNotes($oldID-Position)
	    setIfExists Notes($newID-Color) backupNotes($oldID-Color)
	    drawNotes $newID
	}
	
	foreach xconn $connections {
	    if { $Subnet([oMod xconn]) == $Subnet([iMod xconn]) } continue
	    if [string equal [oPort conn] [oPort xconn]] {
		listFindAndRemove connections $xconn
		set newconn [lreplace $newconn 2 2 [iMod xconn]]
		set newconn [lreplace $newconn 3 3 [iNum xconn]]
		createConnection $newconn 0 0
		set oldID [makeConnID $xconn]
		set newID [makeConnID $newconn]
		setIfExists Notes($newID) backupNotes($oldID)
		setIfExists Notes($newID-Position) backupNotes($oldID-Position)
		setIfExists Notes($newID-Color) backupNotes($oldID-Color)
		drawNotes $newID
	    }
	}
    }
    showSubnetWindow $subnet [subnet_bbox $subnet]
    unselectAll
}



proc expandSubnet { modid } {
    global Subnet CurrentlySelectedModules
    set from $Subnet(${modid}_num)
    set to $Subnet($modid)

    set fromcanvas $Subnet(Subnet${from}_canvas)
    set tocanvas $Subnet(Subnet${to}_canvas)
    set x [lindex [$tocanvas coords $modid] 0]
    set y [lindex [$tocanvas coords $modid] 1]
    set toDelete ""
    set toAdd ""

    foreach iconn $Subnet(Subnet${from}_connections) {
	lappend toDelete $iconn
	foreach econn $Subnet(SubnetIcon${from}_connections) {
	    lappend toDelete $econn
	    if { [iNum econn] == [oNum iconn] &&
		 [string equal SubnetIcon$from [iMod econn]] &&
		 [string equal Subnet$from [oMod iconn]] } {
		lappend toAdd \
		    "[oMod econn] [oNum econn] [iMod iconn] [iNum iconn]"
		
	    }
	    if { [oNum econn] == [iNum iconn] &&
		 [string equal SubnetIcon$from [oMod econn]] &&
		 [string equal Subnet$from [iMod iconn]] } {
		lappend toAdd \
		    "[oMod iconn] [oNum iconn] [iMod econn] [iNum econn]"
	    }
	}
    }

    foreach conn $toDelete {
	# the last 0 paramater means to not tell scirun, just delete TCL reps
	destroyConnection $conn 0 0
    }

    foreach module $Subnet(Subnet${from}_Modules) {
	set bbox [$fromcanvas bbox $module]
	set newx [expr $x + [lindex $bbox 0] - 10]
	set newy [expr $y + [lindex $bbox 1] - 25]
	set Subnet($module) $to
	lappend Subnet(Subnet${to}_Modules) $module
	$fromcanvas delete $module
	destroy $fromcanvas.module$module
	$module make_icon $newx $newy 1
    }	

    foreach conn $toAdd {
	# the last 0 paramater means to not tell scirun, just delete TCL reps
	createConnection $conn 0 0
    }

    # Delete Icon from canvases
    $Subnet(Subnet$Subnet($modid)_canvas) delete $modid
    destroy $Subnet(Subnet$Subnet($modid)_canvas).module$modid
    $Subnet(Subnet$Subnet($modid)_minicanvas) delete $modid
    
    # Remove references to module is various state arrays
    array unset Subnet ${modid}_connections
    listFindAndRemove CurrentlySelectedModules $modid
    listFindAndRemove Subnet(Subnet$Subnet($modid)_Modules) $modid

    set CurrentlySelectedModules $Subnet(Subnet${from}_Modules)
    foreach module $Subnet(Subnet${from}_Modules) {
	drawConnections $Subnet(${module}_connections)
	$module setColorAndTitle
    }

    trace vdelete Subnet(Subnet${from}_Name) w updateSubnetName
    array unset Subnet Subnet${from}*
    array unset Subnet SubnetIcon${from}*
    destroy .subnet$from    
}

# Sorts a list of connections by input port position left to right
proc sortPorts { subnet connections } {
    global Subnet
    set xposlist ""
    for {set i 0} { $i < [llength $connections] } { incr i } {
	set conn [lindex $connections $i]
	if {$Subnet([oMod conn]) == $subnet} {
	    set pos [portCoords [oPort conn]]
	} else {
	    set pos [portCoords [iPort conn]]
	}
	lappend xposlist [list [lindex $pos 0] $i]
    }
    set xposlist [lsort -real -index 0 $xposlist]
    set retval ""
    foreach index $xposlist {
	lappend retval [lindex $connections [lindex $index 1]]
    }
    return $retval
}

proc drawSubnetConnections { subnet } {
    global Subnet
    drawConnections $Subnet(Subnet${subnet}_connections)
}

# This procedure caches away all the global variables that match the pattern
# "mDDDD" (where D is a decimal digit or null character"
# this allow nested source loads of the networks (see the .net files)
proc backupLoadVars { key } {
    global loadVars
    set pattern m
    for {set i 0} {$i < 4} {incr i} {
	set pattern "$pattern\\\[0\\\-9\\\]"
	set varNames [uplevel \#0 info vars $pattern]
	eval lappend {loadVars($key-varList)} $varNames
	foreach name $varNames {
	    upvar \#0 $name var
	    set loadVars($key-$name) $var
	}
    }
}

proc restoreLoadVars { key } {
    global loadVars
    if ![info exists loadVars($key-varList)] { return }
    foreach name $loadVars($key-varList) {
	upvar \#0 $name var
	set var $loadVars($key-$name)
    }
    array unset loadVars "$key-*"
}

proc loadSubnet { filename { x 0 } { y 0 } } {
    global Subnet SCIRUN_SRCDIR Name
    set subnet $Subnet(Loading)
    if {!$x && !$y} {
	global mouseX mouseY
	set x [expr $mouseX+[$Subnet(Subnet${subnet}_canvas) canvasx 0]]
	set y [expr $mouseY+[$Subnet(Subnet${subnet}_canvas) canvasy 0]]
    }
    set splitname [file split $filename]
    set netname [lindex $splitname end]
    if { [llength $splitname] == 1 } {
	set filename [file join $SCIRUN_SRCDIR Subnets $netname]
	if ![file exists $filename] {
	    set filename [file join ~ SCIRun Subnets $netname]
	}
    }

    if { ![file exists $filename] } {
	tk_messageBox -type ok -parent . -icon warning -message \
	    "Subnet file \"$filename\" does not exist."
	return
    }

    set subnetNumber [makeSubnet $subnet $x $y]
    set Subnet(Loading) $subnetNumber
    backupLoadVars $filename
    uplevel \#0 source \{$filename\}
    restoreLoadVars $filename
    set Subnet(Subnet$Subnet(Loading)_filename) "$filename"
    set Subnet(Subnet$Subnet(Loading)_Name) \
	[join [lrange [split [lindex [file split $filename] end] "."] 0 end-1] "."]
    set Subnet(Loading) $subnet
    return SubnetIcon$subnetNumber
}

proc addSubnetInstanceAtPosition { name x y } {
    global Subnet
    set from $Subnet(Loading)    
    set to [makeSubnet $from $x $y]
    set Subnet(Loading) $to
    instantiateSubnet$name
    set Subnet(Loading) $from
    return SubnetIcon$to
}

proc isaDefaultValue { var } {
    set scope [string first :: $var]
    if { $scope == 0 } { set var [string range $var 2 end] }
    # Find string position where Module and Variable name are deliniated by -
    set pos [string first - $var]
    # Get the module instantiations name
    set module [string range $var 0 [expr $pos-1]]
    # Get the variables name
    set varname [string range $var [expr $pos+1] end]
    # Get where in the package hierarchy the module resides
    set modulePath [list [netedit packageName $module] \
			 [netedit categoryName $module] \
			 [netedit moduleName $module]]
    # Get the name of the TCL instance of the modules GUI
    set classname [join $modulePath _]
    if { ![llength [info commands $classname]] } { 
	error "TCL Class not found while writing guiVar $var"	
	return 0 
    }
    # If it doesn't already exist from a previous check...
    set command ${classname}-DEFAULT
    if { ![llength [info commands $command]] } {
	# Then try and create a default TCL instance of that modules GUI
	eval $classname $command
    }
    # If the default variable hasn't been created in TCL yet...
    if { [llength [uplevel \#0 info vars \"$command-$varname\"]] != 1 } { 
	# Assume the variable we're checknig is NOT DEFAULT and return FALSE
	return 0
    }
    # Get the variables at the global level
    upvar \#0 "$command-$varname" tocheck_value "$module-$varname" default_value
    # Compare strings values exactly, returns FALSE if there is any differnce
    return [string equal $tocheck_value $default_value]
}

proc writeSubnets { filename { subnet 0 } } {
    global Subnet
    set Subnet(Subnet${subnet}_instance) [join $Subnet(Subnet${subnet}_Name) ""]
    foreach module $Subnet(Subnet${subnet}_Modules) {
	if [isaSubnetIcon $module] {
	    writeSubnets $filename $Subnet(${module}_num)
	}
    }
    writeSubnet $filename $subnet
}

proc writeSubnet { filename subnet } {
    global Subnet Disabled Notes
    set out [open $filename {WRONLY APPEND}]
    set connections ""
    set modVar(Subnet${subnet}) "Subnet"
    
    if $subnet {
	set tab "   "
	puts $out "proc instantiateSubnet$Subnet(Subnet${subnet}_instance) \{\} \{"
    } else {
	set tab ""
    }
    
    puts $out "${tab}global Subnet"
    puts $out "${tab}set Subnet(Subnet\$Subnet(Loading)_Name) \{$Subnet(Subnet${subnet}_Name)\}"
    puts $out "${tab}set bbox \{[subnet_bbox $subnet]\}"

    set i 0
    foreach module $Subnet(Subnet${subnet}_Modules) {
	set modVar($module) "\$m$i"
	if { [isaSubnetIcon $module] } {
	    puts $out "\n${tab}\# Instiantiate a SCIRun Sub-Network"
	    set number $Subnet(${module}_num)
	    puts -nonewline $out "${tab}set m$i \[addSubnetInstanceAtPosition $Subnet(Subnet${number}_instance) "
	} else {
	    set modulePath [list [netedit packageName $module] \
				 [netedit categoryName $module] \
				 [netedit moduleName $module]]
	    puts $out "\n${tab}\# Create a [join $modulePath ->] Module"
	    puts -nonewline $out  "${tab}set m$i \[addModuleAtPosition "
	    foreach elem $modulePath { puts -nonewline $out "\"${elem}\" " }
	}
	# Write the x,y position of the modules icon on the network graph
	puts $out "[expr int([$module get_x])] [expr int([$module get_y])]\]"
	# Cache all connections to a big list to write out later in the file
	eval lappend connections $Subnet(${module}_connections)
	# Write user notes 
	if [info exists Notes($module-Position)] {
	    puts $out "${tab}set Notes(\$m$i-Position) \{$Notes($module-Position)\}"
	}
	if [info exists Notes($module-Color)] {
	    puts $out "${tab}set Notes(\$m$i--Color) \{$Notes($module-Color)\}"
	}
	incr i
    }

    # Uniquely sort connections list by output port # to handle dynamic ports
    set connections [lsort -integer -index 3 [lsort -unique $connections]]

    if [llength $connections] {
	puts $out "\n${tab}\# Create the Connections between Modules"
    }
    set i 0
    foreach conn $connections {
	puts -nonewline $out "${tab}set c$i \[addConnection "
	puts $out "$modVar([oMod conn]) [oNum conn] $modVar([iMod conn]) [iNum conn]\]"
	set id [makeConnID $conn]
	if [info exists Notes($id-Color)] {
	    puts $out "${tab}set Notes(\$c$i-Color) \{$Notes($id-Color)\}"
	}	
	if { [info exists Disabled($id)] && $Disabled($id) } {
	    puts $out "${tab}set Disabled(\$c$i) \{1\}"
	}
	if { [info exists Notes($id)] && [string length $Notes($id)] } {
	    puts $out "${tab}set Notes(\$c$i) \{$Notes($id)\}"
	}
	if [info exists Notes($id-Position)] {
	    puts $out "${tab}set Notes(\$c$i-Position) \{$Notes($id-Position)\}"
	}
	incr i
    }
    close $out

    set i 0
    foreach module $Subnet(Subnet${subnet}_Modules) {
	if { [isaSubnetIcon $module] } { incr i; continue } 
	set out [open $filename {WRONLY APPEND}] ;# Re-Open for appending
	set modulePath [list [netedit packageName $module] \
			    [netedit categoryName $module] \
			    [netedit moduleName $module]]
	puts $out "\n${tab}\# Set GUI Values for the [join $modulePath ->] Module"

	# Write file to open GUI on load if it was open on save
	if [windowIsMapped .ui$module] {
	    puts $out "${tab}\$m$i initialize_ui"
	}	
	close $out

	# C-side knows which GUI vars to write out
	if { ![isaSubnetIcon $module] } { 
	    $module-c emit_vars $filename "\$m$i" "${tab}"
	}
	incr i
    }   

    if $subnet {
	set out [open $filename {WRONLY APPEND}] ;# Re-Open for appending
	puts $out "\}\n"
	close $out
    }
}		          


proc getAdjWidth { bbox } {
    set minx 90
    set wid [expr [lindex $bbox 2]-[lindex $bbox 0]+$minx/2]
    set maxsize [wm maxsize .]
    set wid [expr $wid>[lindex $maxsize 0]?[lindex $maxsize 0]:$wid]
    set wid [expr $wid<$minx?$minx:$wid]
}


proc getAdjHeight { bbox } {
    set miny 200	
    set hei [expr [lindex $bbox 3]-[lindex $bbox 1]+$miny]
    set maxsize [wm maxsize .]
    set hei [expr $hei>[lindex $maxsize 1]?[lindex $maxsize 1]:$hei]
    set hei [expr $hei<$miny?$miny:$hei]
}
    
    
proc showSubnetWindow { subnet { bbox "" } } {
    wm deiconify .subnet${subnet}
#    raise .subnet${subnet}
    update idletasks
    global Subnet
    if { $bbox == "" } {
	set bbox [subnet_bbox $subnet]
    }
    wm geometry .subnet${subnet} [getAdjWidth $bbox]x[getAdjHeight $bbox]
    set scroll [$Subnet(Subnet${subnet}_canvas) cget -scrollregion]
    $Subnet(Subnet${subnet}_canvas) xview moveto \
	[expr ([lindex $bbox 0]-20)/([lindex $scroll 2]-[lindex $scroll 0])]
    $Subnet(Subnet${subnet}_canvas) yview moveto \
	[expr ([lindex $bbox 1]-20)/([lindex $scroll 3]-[lindex $scroll 1])]
    update idletasks
}
