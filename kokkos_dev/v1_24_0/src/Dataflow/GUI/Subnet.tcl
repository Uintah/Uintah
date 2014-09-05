#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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

    method have_ui {} {
	return [expr ![envBool SCIRUN_DISABLE_SUBNET_UI_BUTTON]]
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

    #  Make the modules icon on a particular canvas
    method make_icon {modx mody { ignore_placement 0 } } {
	global Disabled Subnet Color ToolTipText
	set done_building_icon 0
	set Disabled([modname]) 0
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set minicanvas $Subnet(Subnet$Subnet([modname])_minicanvas)
	
	set modframe $canvas.module[modname]
	frame $modframe -relief raised -borderwidth 3 
	
	bind $modframe <1> "moduleStartDrag [modname] %X %Y 0"
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
	    pack $p.ui -side left -ipadx 5 -ipady 2 -pady 6 -anchor nw
#	    place $p.ui -in $p -x 5 -y 10 -anchor nw
	    Tooltip $p.ui $ToolTipText(ModuleUI)
	}
	# Make the Subnet Button
	if {$isSubnetModule} {
	    global smallIcon
	    button $p.subnet -image $smallIcon -borderwidth 2 \
		-font $ui_font -anchor center -padx 0 -pady  0\
		-width [expr [image width $smallIcon]-2] \
		-height [expr [image height $smallIcon]-5] \
		-command "showSubnetWindow $subnetNumber"
	    pack $p.subnet -side left -padx 3 -ipady 0 -anchor w
#	    place $p.subnet -in $p -x 5 -y 50 -anchor nw

	    Tooltip $p.subnet $ToolTipText(ModuleSubnetBtn)
	}

	# Make the message indicator
	frame $p.msg -relief sunken -height 15 -borderwidth 1 \
	    -width [expr $indicator_width+2]
	pack $p.msg -side right \
	    -padx 2 -pady 2
	frame $p.msg.indicator -relief raised -width 0 -height 0 \
	    -borderwidth 2 -background blue
	bind $p.msg.indicator <Button> "$this displayLog"
	Tooltip $p.msg.indicator $ToolTipText(ModuleMessageBtn)


	# Make the title
	label $p.title -text "$name" -font $modname_font -anchor w
	pack $p.title -side top -padx 2 -anchor w
	bind $p.title <Map> "$this setDone"

	setIfExists instance Subnet(Subnet$Subnet([modname])_Insntace)] ""
	label $p.type -text "$instance" -font $modname_font -anchor w
	pack $p.type -side top -padx 2 -anchor w
	bind $p.type <Map> "$this setDone"

	update_msg_state

	# compute the placement of the module icon
	if { !$ignore_placement } {
	    set pos [findModulePosition $Subnet([modname]) $modx $mody]
	} else {
	    set pos [list $modx $mody]
	}
	
	set pos [eval clampModuleToCanvas $pos]
	
	# Stick it in the canvas
	$canvas create window [lindex $pos 0] [lindex $pos 1] -anchor nw \
	    -window $modframe -tags "module [modname]"

	set pos [scalePath $pos]
	$minicanvas create rectangle [lindex $pos 0] [lindex $pos 1] \
	    [expr [lindex $pos 0]+4] [expr [lindex $pos 1]+2] \
	    -outline "" -fill $Color(Basecolor) -tags "module [modname]"

	# Create, draw, and bind all input and output ports
	drawPorts [modname]
	
	# create the Module Menu
	menu $p.menu -tearoff false -disabledforeground white

	Tooltip $p $ToolTipText(Module)

	bindtags $p [linsert [bindtags $p] 1 $modframe]
	bindtags $p.title [linsert [bindtags $p.title] 1 $modframe]
	bindtags $p.type [linsert [bindtags $p.type] 1 $modframe]

	# If we are NOT currently running a script... ie, not loading the net
	# from a file
	if ![string length [info script]] {
	    unselectAll
	    global CurrentlySelectedModules
	    set CurrentlySelectedModules "[modname]"
	}
	
	fadeinIcon [modname]
    }
    # end make_icon


}
       

# updateSubnetName is called via a trace when the
# Subnet(Subnet${subnet_number}_Name) variable is written
# Automatically updates the Subnet Icon and the Subnet Network Editor names
proc updateSubnetName { subnet_number name1 name2 op } {
    global Subnet
    set Subnet($name2) [join [split $Subnet($name2) \"\{\}] ""]

    # Set the title bar for the Subnet Network Editor Window
    if [winfo exists .subnet${subnet_number}] {
	wm title .subnet${subnet_number} "$Subnet($name2) Sub-Network Editor"
    }
    # Set the title for the Subnet Icon
    if { [llength [info command SubnetIcon${subnet_number}]] } { 
	SubnetIcon${subnet_number} setColorAndTitle
    }
}


# updateSubnetState is called via a trace when the
# Subnet(Subnet${subnet_number}_State) variable is written
proc updateSubnetState { subnet_number name1 name2 op } {
    global Subnet InstanceNames
    set name $Subnet(Subnet${subnet_number}_Name)
    if { [string equal embedded $Subnet(Subnet${subnet_number}_State)] } {
	set Subnet(Subnet${subnet_number}_Instance) [generateInstanceName $name]
    } else {
	if { [info exists InstanceNames($name)] } {
	    set pos [lsearch $InstanceNames($name) $Subnet(Subnet${subnet_number}_Instance)]
	    set InstanceNames($name) [lreplace $InstanceNames($name) $pos $pos]
	}
	set Subnet(Subnet${subnet_number}_Instance) "On Disk"
    }
    SubnetIcon$subnet_number setColorAndTitle
}

proc getAllSubnetNames {} {
    global Subnet SubnetScripts
    set taken ""
    if { [info exists SubnetScripts] } {
	eval lappend taken [array names SubnetScripts]
    }
    foreach namekey [array names Subnet *_Name] {
	lappend taken $Subnet($namekey)
    }
    return $taken
}    

proc generateUniqueSubnetName {} {
    global Subnet SubnetScripts
    set taken [getAllSubnetNames]
    set num 1
    set name "Sub-Network \#$num"
    while { [lsearch $taken $name] != -1 } {
	incr num
	set name "Sub-Network \#$num"
    }
    return $name
}
    
    


# makeSubnetEditorWindow 
# creates:
#  An empty Subnet Editor and a new subnet icon in the subnet $from_subnet
# returns:
#  The newly created Subnet's number
proc makeSubnetEditorWindow { from_subnet x y { bbox "0 0 0 0" } } {
    global Subnet mainCanvasHeight mainCanvasWidth Color

    # Setup default Subnet Variables
    incr Subnet(num)
    set subnet_id				$Subnet(num)
    set Subnet(Subnet${subnet_id})		${subnet_id}
    set Subnet(Subnet${subnet_id}_Name)	    [generateUniqueSubnetName]
    set Subnet(Subnet${subnet_id}_Instance) [generateInstanceName $Subnet(Subnet${subnet_id}_Name)]
    set Subnet(Subnet${subnet_id}_Modules)	""
    set Subnet(Subnet${subnet_id}_connections)	""
    set Subnet(Subnet${subnet_id}_State)	"embedded"
    initInfo $subnet_id

    # Automatically update icon and window title when Subnet name changes
    trace variable Subnet(Subnet${subnet_id}_Name) w \
	"updateSubnetName ${subnet_id}"

    trace variable Subnet(Subnet${subnet_id}_State) w \
	"updateSubnetState ${subnet_id}"

    set w .subnet${subnet_id}
    
    # I don't think condition will ever be true, but be prudent anyways
    if {[winfo exists $w]} { wm deiconfiy $w; raise $w; return }

    # Create the Subnet window and hide it
    toplevel $w -width [getAdjWidth $bbox] -height [getAdjHeight $bbox] 
    wm withdraw $w

    # Set the window title
    wm title $w "$Subnet(Subnet${subnet_id}_Name) Sub-Network Editor"

    # When user clicks the X to delete the window, really just hide it
    wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"
    update idletasks

    # Make the Subnet Menu Bar
    frame $w.main_menu -relief raised -borderwidth 3

    # Make the File menu item
    menubutton $w.main_menu.file -text "File" -underline 0 \
	-menu $w.main_menu.file.menu
    menu $w.main_menu.file.menu -tearoff false
    $w.main_menu.file.menu add command -label "Save As Template..." \
	-underline 0 -command "saveSubnet ${subnet_id} 1"
    $w.main_menu.file.menu add command -label "Save Template..." \
	-underline 0 -command "saveSubnet ${subnet_id} 0"
    $w.main_menu.file.menu add command -label "Network Info..." \
	-underline 0 -command "popupInfoMenu ${subnet_id}"
    pack $w.main_menu.file -side left

    # Make the Edit menu item
    if 0 {
	menubutton $w.main_menu.edit -text "Edit" -underline 0 \
	    -menu $w.main_menu.edit.menu
	menu $w.main_menu.edit.menu -tearoff false
	
	$w.main_menu.edit.menu add command -label "Cut" \
	-underline 0 -command "" -state disabled
	$w.main_menu.edit.menu add command -label "Copy" \
	    -underline 0 -command "" -state disabled
	$w.main_menu.edit.menu add command -label "Paste" \
	    -underline 0 -command "" -state disabled
	
	pack $w.main_menu.edit -side left
    }

    # Make the Packages menu item
    menubutton $w.main_menu.packages -text "Packages" -underline 0 \
	-menu $w.main_menu.packages.menu
    menu $w.main_menu.packages.menu -tearoff false -postcommand \
	"createModulesMenu $w.main_menu.packages.menu $subnet_id"
#    createModulesMenu $w.main_menu.packages.menu $subnet_id
    pack $w.main_menu.packages -side left

    # Make the Subnet Name Entry Field
    frame $w.fname -borderwidth 2
    label $w.fname.label -text "Name"
    entry $w.fname.entry -validate all \
	-textvariable Subnet(Subnet${subnet_id}_Name)
    pack $w.fname.label $w.fname.entry -side left -pady 5

    # Make the Subnet Canvas
    frame $w.can -relief flat -borderwidth 0
    frame $w.can.can -relief sunken -borderwidth 3
    pack $w.can.can -fill both -expand yes -pady 8
    set Subnet(Subnet${subnet_id}_canvas) "$w.can.can.canvas"
    set canvas $Subnet(Subnet${subnet_id}_canvas)
    canvas $canvas -bg $Color(SubnetEditor) -takefocus 1 \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight"
    pack $canvas -expand yes -fill both

    # Create a BOGUS minicanvas for Subnet/Main Editor code compatibility
    # it just never gets packed and therefore not displayed
    set Subnet(Subnet${subnet_id}_minicanvas) $w.can.minicanvas
    canvas $Subnet(Subnet${subnet_id}_minicanvas)
       
    # Make the background square in the canvas to catch all mouse events
    $canvas create rectangle 0 0 $mainCanvasWidth $mainCanvasWidth \
	-fill $Color(SubnetEditor) -tags "bgRect"

    # Create the Canvas Scrollbars
    scrollbar $w.hscroll -relief sunken -orient horizontal \
	-command "$canvas xview"
    scrollbar $w.vscroll -relief sunken -command "$canvas yview" 

    # Configure the Subnet Editor Canvas Scrollbars
    # drawSubnetConnections causes connections to be redrawn on a scroll
    $canvas configure \
	-yscrollcommand "drawSubnetConnections ${subnet_id};$w.vscroll set" \
	-xscrollcommand "drawSubnetConnections ${subnet_id};$w.hscroll set"

    # Make the Subnet State radio buttons
    frame $w.state
    radiobutton $w.state.embedded -text Embedded -value embedded \
	-variable Subnet(Subnet${subnet_id}_State) 
    radiobutton $w.state.ondisk -text "On Disk" -value ondisk \
	-variable Subnet(Subnet${subnet_id}_State)
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

    # Create the Subnet itcl class
    set Subnet(SubnetIcon${subnet_id}) $from_subnet
    SubnetModule SubnetIcon${subnet_id} \
	-name $Subnet(Subnet${subnet_id}_Name) -subnetNumber ${subnet_id}
    set Subnet(SubnetIcon${subnet_id}_num) ${subnet_id}
    set Subnet(SubnetIcon${subnet_id}_connections) ""

    # Make the icon for the new Subnet on the old canvas
    SubnetIcon${subnet_id} make_icon $x $y 1
    lappend Subnet(Subnet${from_subnet}_Modules) SubnetIcon${subnet_id}

    # Keybinding to bring up the Packages Menu
    $canvas bind bgRect <3> "modulesMenu ${subnet_id} %x %y"
    # Keybindings for Select Operations
    $canvas bind bgRect <1> "focus $canvas; startBox $canvas %X %Y 0"
    $canvas bind bgRect <Control-Button-1> "startBox $canvas %X %Y 1"
    $canvas bind bgRect <B1-Motion> "makeBox $canvas %X %Y"
    $canvas bind bgRect <ButtonRelease-1> "$canvas delete tempbox"
    # Redraw the connections when the Subnet Editor window is resized
    bind $canvas <Configure> "drawSubnetConnections ${subnet_id}"
    # Mouse Binding up-down on scroll wheel to move up-down on SubCanvas
    bind $w <ButtonPress-5>  "canvasScroll $canvas 0.0 0.01"
    bind $w <ButtonPress-4>  "canvasScroll $canvas 0.0 -0.01"
    # Bindings for SubCanvas movement on arrow keys press
    bind $w.can.can.canvas <KeyPress-Down>  "canvasScroll $canvas 0.0 0.01"
    bind $w.can.can.canvas <KeyPress-Up>    "canvasScroll $canvas 0.0 -0.01"
    bind $w.can.can.canvas <KeyPress-Left>  "canvasScroll $canvas -0.01 0.0"
    bind $w.can.can.canvas <KeyPress-Right> "canvasScroll $canvas 0.01 0.0"
#    $canvas bind <KeyPress-Right> "canvasScroll $canvas 0.01 0.0" 
    # Other misc. SubCanvas key bindings
    bind $w <Control-d> "moduleDestroySelected"
    bind $w <Control-l> "ClearCanvas 1 ${subnet_id}"
    bind $w <Control-z> "undo"
    bind $w <Control-a> "selectAll ${subnet_id}"
    bind $w <Control-y> "redo"
    return ${subnet_id}
}


# createSubnetFromModules
proc createSubnetFromModules { args } {
    # Dont create an empty subnet
    if { ![llength $args] } return

    set modules $args
    global Subnet Notes

    # Figure out what subnet we are taking the modules from
    set from_subnet $Subnet([lindex $modules 0])

    # Verify that all the modules are all from the same subnet
    foreach modid $modules {
	if { $from_subnet != $Subnet($modid) } {
	    displayErrorWarningOrInfo \ 
	    "*** $modid does not exist in subnet level $from_subnet" error
	    return
	}
    }
    
    # Delete the module icons from the old canvas, 
    # also, create a list of connections that need to be deleted
    set connections ""
    foreach modid $modules {
	listFindAndRemove Subnet(Subnet${from_subnet}_Modules) $modid
	eval lappend connections $Subnet(${modid}_connections)
    }

    # remove all duplicate connections from the list
    set connections [lsort -unique $connections]

    # delete connections in decreasing input port number for dynamic ports
    foreach conn [lsort -decreasing -integer -index 3 $connections] {
	set id [makeConnID $conn]
	setIfExists backupNotes($id) Notes($id) 
	setIfExists backupNotes($id-Position) Notes($id-Position) 
	setIfExists backupNotes($id-Color) Notes($id-Color) 
	destroyConnection $conn 0 0
    }    

    set from_canvas $Subnet(Subnet${from_subnet}_canvas)    
    set bbox [compute_bbox $from_canvas $modules]
    set x [lindex $bbox 0]
    set y [lindex $bbox 1]

    # Create the empty Subnet Network Editor and its Icon
    set subnet_number [makeSubnetEditorWindow $from_subnet $x $y $bbox]
    set Subnet(Subnet${subnet_number}_Modules) $modules

    # Move each module to the new canvas
    foreach modid $modules {
	# Get old module's icon position before deleting it
	set modbbox [$from_canvas bbox $modid]
	# Delete the icon on the from canvases
	$from_canvas delete $modid $modid-notes $modid-notes-shadow
	destroy $from_canvas.module$modid
	$Subnet(Subnet$Subnet($modid)_minicanvas) delete $modid
	# Compute position of the icon on the new canvas
	set newx [expr [lindex $modbbox 0] - $x + 10]
	set newy [expr [lindex $modbbox 1] - $y + 25]
	set Subnet($modid) $subnet_number
	# Create icon on new canvas
	$modid make_icon $newx $newy 1
    }
    update idletasks
    # Create new connections to Subnet Icon and within Subnet Editor
    # Note the 0 0 parameters to createConnection:  SCIRun wont be
    # notified of any creation or deletion of connections between modules
    # since no actual connections are being changed
    # sort the connections by port's horizontal location
    set connections [sortPorts $subnet_number $connections]
    set notesList ""
    while { [llength $connections] } {
	# pop the connection off the end of the list
	set conn [lindex $connections 0]
	set connections [lrange $connections 1 end]
	
	# If the connection is contained entirely within the new subnet
	# it doesn't need any extra external connections
	if { $Subnet([oMod conn]) == $Subnet([iMod conn]) } {
	    createConnection $conn 0 0
	    lappend notesList "[makeConnID $conn] [makeConnID $conn]"
	    continue
	}

	if { $Subnet([oMod conn]) == ${subnet_number} } {
	    # Split the connection across the subnet boundary
	    set which [portCount "Subnet${subnet_number} 0 i"]
	    set newconn "[oMod conn] [oNum conn] Subnet${subnet_number} $which"
	    createConnection $newconn 0 0	    
	    set newconn "SubnetIcon${subnet_number} $which [iMod conn] [iNum conn]"
	    createConnection $newconn 0 0
	    lappend notesList "[makeConnID $conn] [makeConnID $newconn]"
	} elseif { $Subnet([iMod conn]) == ${subnet_number} } {
	    # Split the connection across the subnet boundary
	    set which [portCount "Subnet${subnet_number} 0 o"]
	    set newconn "[oMod conn] [oNum conn] SubnetIcon${subnet_number} $which"
	    createConnection $newconn 0 0
	    set newconn "Subnet${subnet_number} $which [iMod conn] [iNum conn]"
	    createConnection $newconn 0 0
	    lappend notesList "[makeConnID $conn] [makeConnID $newconn]"
	}
	
	# Find all other connections out of this port to modules on other
	# side of Subnet interface and share the connection
	foreach xconn $connections {
	    if { $Subnet([oMod xconn]) == $Subnet([iMod xconn]) } continue
	    if [string equal [oPort conn] [oPort xconn]] {
		listFindAndRemove connections $xconn
		set newconn [lreplace $newconn 2 2 [iMod xconn]]
		set newconn [lreplace $newconn 3 3 [iNum xconn]]
		createConnection $newconn 0 0
		lappend notesList "[makeConnID $xconn] [makeConnID $newconn]"
	    }
	}            
    }
    
    # copy over the notes to the new connections
    foreach oldnewid $notesList {
	set oldID [lindex $oldnewid 0]
	set newID [lindex $oldnewid 1]
	setIfExists Notes($newID) backupNotes($oldID)
	setIfExists Notes($newID-Position) backupNotes($oldID-Position)
	setIfExists Notes($newID-Color) backupNotes($oldID-Color)
	drawNotes $newID
    }

    # Put the window on top and resize it to good dimensions
    showSubnetWindow $subnet_number [subnet_bbox $subnet_number]
    unselectAll
}


# expandSubnet will replace the Subnet icon with its internal modules
# and delete the Subnet Editor associated with that subnet
# Its basically the reverse of createSubnetFromModules
proc expandSubnet { modid } {
    global Subnet
    set from $Subnet(${modid}_num)
    set to $Subnet($modid)

    # create a list of all connections that cross the Subnet Interface
    set toAdd ""
    foreach in $Subnet(Subnet${from}_connections) {
	foreach ex $Subnet(SubnetIcon${from}_connections) {
	    if { [iNum ex] == [oNum in] &&
		 [string equal SubnetIcon$from [iMod ex]] &&
		 [string equal Subnet$from [oMod in]] } {
		# add collapsed connection without Subnet Interface
		lappend toAdd "[oMod ex] [oNum ex] [iMod in] [iNum in]"
	    }
	    if { [oNum ex] == [iNum in] &&
		 [string equal SubnetIcon$from [oMod ex]] &&
		 [string equal Subnet$from [iMod in]] } {
		# add collapsed connection without Subnet Interface
		lappend toAdd "[oMod in] [oNum in] [iMod ex] [iNum ex]"
	    }
	}
    }

    # Delete all connections in Subnet connecting to interface
    # This will cause the external connections to be automatically deleted.
    # While loop evaluates list each time allowing for moved connections
    while { [llength $Subnet(Subnet${from}_connections)] } {
	destroyConnection [lindex $Subnet(Subnet${from}_connections) 0] 0 0
    }


    # Move each module from the Subnet Editor to its containing net
    set x [lindex [$Subnet(Subnet${to}_canvas) coords $modid] 0]
    set y [lindex [$Subnet(Subnet${to}_canvas) coords $modid] 1]
    foreach module $Subnet(Subnet${from}_Modules) {
	set bbox [$Subnet(Subnet${from}_canvas) bbox $module]
	set newx [expr $x + [lindex $bbox 0] - 10]
	set newy [expr $y + [lindex $bbox 1] - 25]
	set Subnet($module) $to
	lappend Subnet(Subnet${to}_Modules) $module
	$module make_icon $newx $newy 1
    }	
    
    # Create the connections between the modules that were just moved
    # from the subnet and the modules that were connecting to the subnet
    foreach conn $toAdd {
	createConnection $conn 0 0 ;# 0 0 = dont tell SCIRun & dont record undo
    }

    # Just select the modules that came in from in the expanded subnet
    unselectAll
    foreach module $Subnet(Subnet${from}_Modules) {
	$module addSelected
    }

    # Delete the Subnet Icon from canvases
    $Subnet(Subnet${to}_canvas) delete $modid
    $Subnet(Subnet${to}_minicanvas) delete $modid
    destroy $Subnet(Subnet${to}_canvas).module$modid
    
    # Destroy the Subnet Editor Window
    destroy SubnetIcon$from    

    # Remove the trace when the Subnet name is changed
    trace vdelete Subnet(Subnet${from}_Name) w updateSubnetName

    # Remove refrences to subnet instance from Subnet global array
    listFindAndRemove Subnet(Subnet${to}_Modules) $modid
    array unset Subnet ${modid}_connections
    array unset Subnet Subnet${from}*
    array unset Subnet SubnetIcon${from}*    
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
    foreach pattern "m c" {
	for {set i 0} {$i < 4} {incr i} {
	    set pattern "$pattern\\\[0\\\-9\\\]"
	    set varNames [uplevel \#0 info vars $pattern]
	    if { ![llength $varNames] } continue
	    eval lappend {loadVars($key-varList)} $varNames
	    foreach name $varNames {
		upvar \#0 $name var
		set loadVars($key-$name) $var
	    }
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

proc loadSubnetFromDisk { name { x 0 } { y 0 } } {
    return [loadSubnet $name $x $y]
}

proc loadSubnet { filename { x 0 } { y 0 } } {
    global Subnet Name
    set subnet $Subnet(Loading)
    if {!$x && !$y} {
	global mouseX mouseY
	set x [expr $mouseX+[$Subnet(Subnet${subnet}_canvas) canvasx 0]]
	set y [expr $mouseY+[$Subnet(Subnet${subnet}_canvas) canvasy 0]]
    }
    set splitname [file split $filename]
    set netname [lindex $splitname end]
    if { [string last .net $netname] != [expr [string length $netname]-4] } {
	set netname ${netname}.net
    }
    if { [llength $splitname] == 1 } {
	set filename [file join [netedit getenv SCIRUN_SRCDIR] Subnets $netname]
	if ![file exists $filename] {
	    set filename [file join ~ SCIRun Subnets $netname]
	}
    }

    if { ![file exists $filename] } {
	tk_messageBox -type ok -parent . -icon warning -message \
	    "Subnet file \"$filename\" does not exist."
	return doNothing
    }

    set subnetNumber [makeSubnetEditorWindow $subnet $x $y]
    set Subnet(Loading) $subnetNumber
    backupLoadVars $filename
    uplevel \#0 "source \{$filename\}"
    restoreLoadVars $filename
    set Subnet(Subnet$Subnet(Loading)_filename) "$filename"
    set Subnet(Subnet$Subnet(Loading)_State) ondisk
    set Subnet(Subnet$Subnet(Loading)_Name) \
	[join [lrange [split [lindex [file split $filename] end] "."] 0 end-1] "."]

    set Subnet(Loading) $subnet

    return SubnetIcon$subnetNumber
}

proc generateInstanceName { name } {
    global InstanceNames
    set i 1
    set retval "Instance \#1"
    if { [info exists InstanceNames($name)] } {
	while { [lsearch $InstanceNames($name) $retval] != -1 } {
	    incr i
	    set retval "Instance \#$i"
	}
    }
    lappend InstanceNames($name) $retval
    return $retval
}

proc instanceSubnet { subnet_name { x 0 } { y 0 } { from 0 } } {
    global Subnet Notes Disabled SubnetScripts name
    if {!$x && !$y} {
	global mouseX mouseY
	set x [expr $mouseX+[$Subnet(Subnet${from}_canvas) canvasx 0]]
	set y [expr $mouseY+[$Subnet(Subnet${from}_canvas) canvasy 0]]
	set mouseX 10
	set mouseY 10
    }
    if { $from == 0 } {
	set from $Subnet(Loading)
    }
    set to [makeSubnetEditorWindow $from $x $y]
    set Subnet(Loading) $to
    backupLoadVars $subnet_name
    uplevel \#0 $SubnetScripts($subnet_name)
    restoreLoadVars $subnet_name
    set Subnet(Subnet${to}_Name) "$subnet_name"
    set Subnet(Subnet${to}_Instance) [generateInstanceName $subnet_name]
    SubnetIcon$to setColorAndTitle
    set Subnet(Loading) $from
    return SubnetIcon$to
}


proc isaDefaultValue { module varname classname } {
    global DefaultClassInstances
    upvar \#0 "$module-$varname" val
    # If this variable doesn't exist in the TCL namespace yet,
    # it was created by a C-side GuiVar, but it hasn't been modified yet,
    # Assume the variable we're checking is DEFAULT, so return TRUE
    if { ![info exists val] } {
	return 1
    }

    # If the default class has not already been instantced
    if { ![info exists DefaultClassInstances($classname)] } {
	if { ![llength [info commands $classname]] } { 
	    error "TCL Class not found while writing guiVar $var"	
	    return 0 
	}
	eval $classname DEFAULT-$classname
	# Save a list of all default classes made for speed
	set DefaultClassInstances($classname) 1
    }

    # Get a link to the newly created default variable at the global level
    upvar \#0 "DEFAULT-$classname-$varname" default_value

    # If the default variable hasn't been created in TCL yet...
    if { ![info exists default_value] } {
	# Assume the variable we're checking is NOT DEFAULT so return FALSE
	return 0
    }

    # Some sort of default variable was created, so we
    # Compare strings values exactly, and return FALSE if there is a differnce
    return [string equal $default_value $val]
}

proc writeSubnetOnDisk { id } {
    global Subnet
    set filename [file join ~ SCIRun Subnets $Subnet(Subnet${id}_Name).net]
    if { [info exists Subnet(Subnet${id}_Filename)] } {
	set filename $Subnet(Subnet${id}_Filename)
    }
    set dir [lrange [file split $filename] 0 end-1]
    set dir [eval file join $dir]
    catch "file mkdir $dir"
    if { [validDir $dir] } {
	writeNetwork $filename $id
    }
}



proc writeSubnets { file subnet_ids } {
    global Subnet SubnetScripts
    set alreadyWrittenToScript ""
    set alreadyWrittenToDisk ""
    while { [llength $subnet_ids] } {
	set id [popFront subnet_ids]
	foreach module $Subnet(Subnet${id}_Modules) {
	    if { ![isaSubnetIcon $module] } continue
	    set sub_id $Subnet(${module}_num)
	    set subname $Subnet(Subnet${sub_id}_Name)
	    if { [string equal $Subnet(Subnet${sub_id}_State) "ondisk"] } {
		if { ($sub_id != 0) && \
		     ([lsearch $alreadyWrittenToDisk $subname] == -1) } {
		    lappend alreadyWrittenToDisk $subname
		    
		    writeSubnetOnDisk $sub_id
		}
	    } elseif { [lsearch $alreadyWrittenToScript $subname] == -1 } {
		lappend subnet_ids $sub_id
		lappend alreadyWrittenToScript $subname

		set SubnetScripts($subname) [genSubnetScript $sub_id]
		puts -nonewline $file "addSubnetToDatabase \{"
		puts -nonewline $file $SubnetScripts($subname)
		puts $file "\}\n"
	    }
	}
    }
}


proc doNothing { args } {
}

proc counting_addModuleAtPosition { args } {
    global scriptCount
    incr scriptCount(Total) 6
    incr scriptCount(addModuleAtPosition)
    return doNothing
}

proc counting_addConnection { args } {
    global scriptCount
    incr scriptCount(Total) 4
    incr scriptCount(addConnection)
    return doNothing
}


proc counting_set { args } {
    global scriptCount
    if { [info level]==$scriptCount(setLevel) } { 
	incr scriptCount(Total)
	incr scriptCount(set)
    }
    return [uplevel 1 real_set $args]
}

proc counting_instanceSubnet { args } {
    global scriptCount
    incr scriptCount(Total) 10
    return doNothing
}

proc counting_loadSubnetFromDisk { args } {
    global scriptCount
    incr scriptCount(Total) 10
    return doNothing
}


proc counting_addSubnetToDatabase { args } {
    global scriptCount
    incr scriptCount(Total)
}


proc loading_addModuleAtPosition { args } {
    global PowerApp progressMeter
    if { [info exists progressMeter] } {
	if { !$PowerApp } {
	    set modulepath [join [lrange $args 0 2] ->]
	    setProgressText "Loading Module: $modulepath"
	}
	incrProgress 6
    }

    return [uplevel 1 real_addModuleAtPosition $args]
}

proc loading_addConnection { args } {
    global scriptCount PowerApp progressMeter
    if { [info exists progressMeter] } {
	if { !$PowerApp } {
	    incr scriptCount(addConnectionLoading)
	    setProgressText "Creating connection \# $scriptCount(addConnectionLoading) of $scriptCount(addConnection)"
	    if { $scriptCount(addConnectionLoading) == $scriptCount(addConnection) } {
		setProgressText "Loading module GUI settings..."
	    }
	}
	incrProgress 3
    }
    return [uplevel 1 real_addConnection $args]
}

proc loading_set { args } {
    if { [info level] == 1 } incrProgress
    return [uplevel 1 real_set $args]
}

proc loading_instanceSubnet { args } {    
    incrProgress 10
    global progressMeter scriptCount
    setProgressText "Creating [lindex $args 0] Sub-Network"
    setIfExists scriptCount(progressMeter) progressMeter
    unsetIfExists progressMeter
    set retval [uplevel 1 real_instanceSubnet $args]
    setIfExists progressMeter scriptCount(progressMeter)
    
    return $retval
}


proc loading_loadSubnetFromDisk { args } {    
    incrProgress 10
    global progressMeter scriptCount
    setProgressText "Creating [lindex $args 0] Sub-Network"
    setIfExists scriptCount(progressMeter) progressMeter
    unsetIfExists progressMeter
    set retval [uplevel 1 real_loadSubnetFromDisk $args]
    setIfExists progressMeter scriptCount(progressMeter)
    
    return $retval
}


proc loading_addSubnetToDatabase { args } {
    incrProgress
    return [uplevel 1 real_addSubnetToDatabase $args]
}



proc renameNetworkCommands { prefix } {
    lappend commands set
    lappend commands addModuleAtPosition
    lappend commands addConnection
    lappend commands instanceSubnet
    lappend commands addSubnetToDatabase
    lappend commands loadSubnetFromDisk

    foreach command $commands {
	set exists [expr [llength [info commands ${prefix}${command}]] == 1]
	set cached [expr [llength [info commands real_${command}]] == 1]
	if { $exists && !$cached } {
	    rename ${command} real_${command}
	    rename ${prefix}${command} ${command}
	} elseif { !$exists && $cached } {
	    rename ${command} ${prefix}${command}
	    rename real_${command} ${command}
	} else {
	    puts "renameNetworkCommands already cached command: $command"
	}
    }
}

proc resetScriptCount {} {
    setGlobal scriptCount(Total) 0
    setGlobal scriptCount(addModuleAtPosition) 0
    setGlobal scriptCount(addConnection) 0
    setGlobal scriptCount(addConnectionLoading) 0
    setGlobal scriptCount(sourceSettingsFile) 0
    setGlobal scriptCount(set) 0
    setGlobal scriptCount(setLevel) [info level]
    
}
    

proc addSubnetToDatabase { script } {
    set testing [interp create -safe]    
    foreach line [split $script "\n"] {
	catch "$testing eval \{$line\}"
	if { [$testing eval info exists Name] } {
	    set name [$testing eval set Name]
	} elseif { [$testing eval info exists name] } {
	    set name [$testing eval set name]
	}
    }
    interp delete $testing

    if { [info exists name] } {
	setGlobal SubnetScripts($name) $script
    }
}


proc subDATADIRandDATASET { val } {
    if { ![envBool SCIRUN_NET_SUBSTITUTE_DATADIR] } { return $val }
    set tmpval $val
    set tmp [netedit getenv SCIRUN_DATA]
    if { [string length $tmp] } {
	set first [string first $tmp $tmpval]
	set last [expr $first+[string length $tmp]-1]
	if { $first != -1 } {
	    set tmpval [string replace $tmpval $first $last "\$DATADIR"]
	}
    }

    set tmp [netedit getenv SCIRUN_DATAFILE]
    if { [string length $tmp] } {
	set first [string first $tmp $tmpval]
	set last [expr $first+[string length $tmp]-1]
	if { $first != -1 } {
	    set tmpval [string replace $tmpval $first $last "\$DATAFILE"]
	}
    }

    set tmp [netedit getenv SCIRUN_DATASET]
    if { [string length $tmp] } {
	set first [string first $tmp $tmpval]
	while { $first != -1 } {
	    set last [expr $first+[string length $tmp]-1]
	    set tmpval [string replace $tmpval $first $last "\$DATASET"]
	    set first [string first $tmp $tmpval]
	}
    }
    return $tmpval
}


proc genSubnetScript { subnet { tab "__auto__" }  } {
    netedit presave

    global Subnet Disabled Notes
    set connections ""
    set modVar(Subnet${subnet}) "Subnet"

    if { [string equal $tab "__auto__"] } {
	if $subnet { set tab "   " } else { set tab "" }
    }
    
    append script "\n${tab}set name \{$Subnet(Subnet${subnet}_Name)\}\n"
    append script "${tab}set bbox \{[subnet_bbox $subnet]\}\n"
    append script "${tab}set creationDate \{$Subnet(Subnet${subnet}_creationDate)\}\n"
    append script "${tab}set creationTime \{$Subnet(Subnet${subnet}_creationTime)\}\n"
    append script "${tab}set runDate \{$Subnet(Subnet${subnet}_runDate)\}\n"
    append script "${tab}set runTime \{$Subnet(Subnet${subnet}_runTime)\}\n"
    append script "${tab}set notes \{$Subnet(Subnet${subnet}_notes)\}\n"
    
    set i 0
    foreach module $Subnet(Subnet${subnet}_Modules) {
	incr i
	set modVar($module) "\$m$i"
	append script "\n"
	if { [isaSubnetIcon $module] } {
	    set number $Subnet(${module}_num)
	    set name $Subnet(Subnet${number}_Name)
	    if { $Subnet(Subnet${number}_State) == "ondisk" } {
		if { [info exists Subnet(Subnet${number}_Filename)] } {
		    set name $Subnet(Subnet${number}_Filename)
		}
		append script "${tab}\# Load $name Sub-Network from disk\n"
		append script "${tab}set m$i \[loadSubnetFromDisk \"${name}\" "

	    } else {
		append script "${tab}\# Create an instance of a $name Sub-Network\n"
		append script "${tab}set m$i \[instanceSubnet \"${name}\" "
	    }
	} else {
	    set modpath [modulePath $module]	    
	    append script "${tab}\# Create a [join $modpath ->] Module\n"
	    append script "${tab}set m$i \[addModuleAtPosition "
	    foreach elem $modpath { append script "\"${elem}\" " }
	}
	# Write the x,y position of the modules icon on the network graph
	append script "[expr int([$module get_x])] [expr int([$module get_y])]\]\n"
	# Cache all connections to a big list to write out later in the file
	eval lappend connections $Subnet(${module}_connections)
	# Write user notes 
	if { [info exists Notes($module)]&&[string length $Notes($module)] } {
	    append script "${tab}set Notes(\$m$i) \{$Notes($module)\}\n"
	    if { [info exists Notes($module-Position)] } {
		append script "${tab}set Notes(\$m$i-Position) "
		append script  "\{$Notes($module-Position)\}\n"
	    }
	    if { [info exists Notes($module-Color)] } {
		append script "${tab}set Notes(\$m$i-Color) "
		append script "\{$Notes($module-Color)\}\n"
	    }
	}
    }

    # Uniquely sort connections list by output port # to handle dynamic ports
    set connections [lsort -integer -index 3 [lsort -unique $connections]]

    if [llength $connections] {
	append script "\n"
	append script "${tab}\# Create the Connections between Modules\n"
    }
    set i 0
    foreach conn $connections {
	incr i
	if {![info exists modVar([oMod conn])] } {
	    puts "Connection output module: [oMod conn] does not exist.\n$conn"
	    continue
	}
	if {![info exists modVar([iMod conn])] } {
	    puts "Connection input module: [iMod conn] does not exist.\n$conn"
	    continue
	}

        append script "${tab}set c$i \[addConnection "
	append script "$modVar([oMod conn]) [oNum conn]"
	append script " $modVar([iMod conn]) [iNum conn]\]\n"
	set id [makeConnID $conn]
	if { [info exists Disabled($id)] && $Disabled($id) } {
	    append script "${tab}set Disabled(\$c$i) \{1\}\n"
	}

	if { [info exists Notes($id)] && [string length $Notes($id)] } {
	    append script "${tab}set Notes(\$c$i) \{$Notes($id)\}\n"
	    if [info exists Notes($id-Position)] {
		append script "${tab}set Notes(\$c$i-Position) "
		append script "\{$Notes($id-Position)\}\n"
	    }
	    if { [info exists Notes($id-Color)] } {
		append script "${tab}set Notes(\$c$i-Color) "
		append script "\{$Notes($id-Color)\}\n"
	    }
	}
    }
    
    set i 0
    foreach module $Subnet(Subnet${subnet}_Modules) {
	$module writeStateToScript script "\$m[incr i]" $tab
    }

    return $script
}


# Get where in the package hierarchy the module resides
proc modulePath { module } {
    return [list [netedit packageName $module] \
		[netedit categoryName $module] \
		[netedit moduleName $module]]
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
    raise .subnet${subnet}
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


proc saveSubnetDialog { subnet_id ask } {
    global Subnet
    if { ![info exists Subnet(Subnet${subnet_id}_Filename)] || \
	     ![string length $Subnet(Subnet${subnet_id}_Filename)]} {
	set ask 1
    }
    if { $ask } {
	set types {
	    {{SCIRun Net} {.net} }
	    {{Other} { * } }
	} 
	set Subnet(Subnet${subnet_id}_Filename) \
	    [tk_getSaveFile -defaultextension {.net} -filetypes $types \
		 -initialdir "[netedit getenv HOME]/SCIRun/Subnets"]
    }
    if { [string length $Subnet(Subnet${subnet_id}_Filename)]} {
	writeNetwork $Subnet(Subnet${subnet_id}_Filename) $subnet_id
    }
}
    
proc loadSubnetScriptsFromDisk { } {
    global SubnetScripts
    set files [glob -nocomplain "[netedit getenv SCIRUN_SRCDIR]/Subnets/*.net"]
    eval lappend files [glob -nocomplain "[netedit getenv HOME]/SCIRun/Subnets/*.net"]
    foreach file $files {
	set script ""
	set handle [open $file RDONLY]
	while { ![eof $handle] } { 
	    append script "[gets $handle]\n"
	}
	close $handle
	addSubnetToDatabase $script
    }
}

proc generateSubnetScriptsFromNetwork { } {
    global Subnet
    for {set i 1} {$i <= $Subnet(num)} {incr i} {
	addSubnetToDatabase [genSubnetScript $i]
    }    
}
