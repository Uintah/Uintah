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

proc setSubnetName { subnet name } {
    global Subnet
    set Subnet(Subnet${subnet}_Name) $name 
    return 1
}

proc makeSubnet { from_subnet x y { bbox "0 0 0 0" }} {
    global Subnet mainCanvasHeight mainCanvasWidth Color
    incr Subnet(num)
    set Subnet(Subnet$Subnet(num))		$Subnet(num)
    set Subnet(Subnet$Subnet(num)_Name)		"Sub-Network \#$Subnet(num)"
    set Subnet(Subnet$Subnet(num)_Modules)	""
    set Subnet(Subnet$Subnet(num)_connections)	""

    trace variable Subnet(Subnet$Subnet(num)_Name) w \
	"updateSubnetName $Subnet(num)"

    set w .subnet$Subnet(num)
    if {[winfo exists $w]} { destroy $w }

    toplevel $w -width [getAdjWidth $bbox] -height [getAdjHeight $bbox] 
    wm withdraw $w
    update idletasks
    wm title $w "$Subnet(Subnet$Subnet(num)_Name) Sub-Network Editor"
    wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"
    
    frame $w.can -relief flat -borderwidth 0
    frame $w.can.can -relief sunken -borderwidth 3
    pack $w.can.can -fill both -expand yes -pady 8

    set Subnet(Subnet$Subnet(num)_canvas) "$w.can.can.canvas"
    set canvas $Subnet(Subnet$Subnet(num)_canvas)

    canvas $canvas -bg $Color(SubnetEditor) \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight"
    $canvas create rectangle -1 -1 $mainCanvasWidth $mainCanvasWidth \
	-fill $Color(SubnetEditor) -tags "bgRect"
    pack $canvas -expand yes -fill both
    menu $canvas.modulesMenu -tearoff false
    
    frame $w.fname -borderwidth 2
    label $w.fname.label -text "Name"
    entry $w.fname.entry -validate all -textvariable Subnet(Subnet$Subnet(num)_Name)
    #	-validatecommand "setSubnetName $Subnet(num) \"%P\""
    pack $w.fname.label $w.fname.entry -side left -pady 5

    frame $w.buttons -borderwidth 2
    button $w.buttons.save -text "Save" -command "saveSubnet $Subnet(num)"
    button $w.buttons.close -text "Close" -command "wm withdraw $w"
    pack $w.buttons.save $w.buttons.close -side left -pady 5 -padx 20
        
    scrollbar $w.hscroll -relief sunken -orient horizontal \
	-command "$canvas xview"
    scrollbar $w.vscroll -relief sunken -command "$canvas yview" 

    $canvas configure \
	-yscrollcommand "drawSubnetConnections $Subnet(num);$w.vscroll set" \
	-xscrollcommand "drawSubnetConnections $Subnet(num);$w.hscroll set"
    grid $w.can $w.hscroll $w.vscroll $w.fname $w.buttons
	
    grid columnconfigure $w 0 -weight 1
    grid rowconfigure    $w 0 -weight 0 
    grid rowconfigure    $w 1 -weight 1 
    grid rowconfigure    $w 2 -weight 0
    grid rowconfigure    $w 3 -weight 0

    grid config $w.fname -column 0 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "snew" 
    grid config $w.can -column 0 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky "snew" 
    grid config $w.hscroll -column 0 -row 2 \
	    -columnspan 1 -rowspan 1 -sticky "ew" -pady 2
    grid config $w.vscroll -column 1 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky "sn" -padx 2
    grid config $w.buttons -column 0 -row 3 \
	    -columnspan 2 -rowspan 1 -sticky "sn" -padx 2


    set Subnet(Subnet$Subnet(num)_minicanvas) $w.can.minicanvas
    canvas $Subnet(Subnet$Subnet(num)_minicanvas)

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
    return $Subnet(num)
}



proc createSubnet { from_subnet { modules "" } } {
    global Subnet CurrentlySelectedModules mouseX mouseY
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
	$canvas delete $modid
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
	set conn [lindex $connections 0]
	set connections [lrange $connections 1 end]
	if { $Subnet([oMod conn]) == $Subnet([iMod conn]) } {
	    createConnection $conn 0 0
	} elseif { $Subnet([oMod conn]) == ${subnet} } {
	    set which [portCount "Subnet${subnet} 0 i"]
	    createConnection \
		"[oMod conn] [oNum conn] Subnet${subnet} $which" 0 0
	    createConnection \
		"SubnetIcon${subnet} $which [iMod conn] [iNum conn]" 0 0
	    foreach xconn $connections {
		if { [oNum conn] == [oNum xconn] &&
		     [string equal [oMod conn] [oMod xconn]] } {
		    listFindAndRemove connections $xconn
		    createConnection \
			"SubnetIcon${subnet} $which [iMod xconn] [iNum xconn]" 0 0
		}
	    }
	} elseif { $Subnet([iMod conn]) == ${subnet} } {
	    set which [portCount "Subnet${subnet} 0 o"]
	    createConnection \
		"Subnet${subnet} $which [iMod conn] [iNum conn]" 0 0
	    createConnection \
		"[oMod conn] [oNum conn] SubnetIcon${subnet} $which" 0 0
	    foreach xconn $connections {
		if { [oNum conn] == [oNum xconn] &&
		     [string equal [oMod conn] [oMod xconn]] } {
		    listFindAndRemove connections $xconn
		    createConnection \
			"Subnet${subnet} $which [iMod xconn] [iNum xconn]" 0 0
		}
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
    if [info exists Name] {
	set Subnet(Subnet$Subnet(Loading)_Name) $Name
    } else {
	set Subnet(Subnet$Subnet(Loading)_Name) [lindex [file split $filename] end]
    }
    set Subnet(Loading) $subnet
    return SubnetIcon$subnetNumber
}


proc saveSubnet { subnet } {
    global Subnet SCIRUN_SRCDIR
    set name [join [split $Subnet(Subnet${subnet}_Name) "/"] ""].net
    set dir [file join $SCIRUN_SRCDIR Subnets]
    catch "file mkdir $dir"
    if ![validDir $dir] {
	set home [file nativename ~]
	set dir [file join $home SCIRun Subnets]
	catch "file mkdir $dir"
    }
    set name [file join $dir $name]
    set Subnet(Subnet${subnet}_filename) $name
    if ![file writable $dir] {
	tk_messageBox -type ok -parent . -icon error -message \
	    "Cannot save Sub-Network $Subnet(Subnet${subnet}_Name) with filename $name to $SCIRUN_SRCDIR/Subnets or $home/SCIRun/Subnets" 
	return
    }
    netedit savenetwork $name $subnet
    return $name
}


proc modVarName { filename modid } {
    global modVar
    set token "$filename-$modid"
    if [info exists modVar($token)] {return $modVar($token)} else { return "" }
}

proc writeSubnetModulesAndConnections { filename {subnet 0}} {
    global Subnet SCIRUN_SRCDIR modVar Disabled Notes
    set out [open $filename {WRONLY APPEND}]

    puts $out "set bbox \{[subnet_bbox $subnet]\}"
    puts -nonewline $out "set Name \{"
    puts $out "$Subnet(Subnet${subnet}_Name)\}"
    set i 0
    set connections ""
    set modVar($filename-Subnet${subnet}) "Subnet"
    puts $out "\n\# Create the Modules"
    foreach module $Subnet(Subnet${subnet}_Modules) {
	if { [isaSubnetIcon $module] } {
	    set iconsubnet $Subnet(${module}_num)
	    # this will always save subnets and set $Subnet(Subnet${iconsubnet}_filename)
	    #[string length $Subnet(Subnet${iconsubnet}_filename)
	    
	    if ![string length [saveSubnet $iconsubnet]] {
		#dont save this module if we cant save the subnet
		continue
	    }
	    set modVar($filename-$module) "\$m$i"
	    puts -nonewline $out "set m$i \[loadSubnet "
	    set splitname [file split $Subnet(Subnet${iconsubnet}_filename)] 
	    puts -nonewline $out "\"[lindex $splitname end]\" "
	} else {
	    set modVar($filename-$module) "\$m$i"
	    puts -nonewline $out  "set m$i \[addModuleAtPosition "
	    puts -nonewline $out  "\"[netedit packageName $module]\" "
	    puts -nonewline $out  "\"[netedit categoryName $module]\" "
	    puts -nonewline $out  "\"[netedit moduleName $module]\" "
	}
	puts $out "[expr int([$module get_x])] [expr int([$module get_y])]\]"
	eval lappend connections $Subnet(${module}_connections)
	incr i
    }
    puts $out "\n\# Set the Module Notes Dispaly Options"
    foreach module $Subnet(Subnet${subnet}_Modules) {
	if ![info exists modVar($filename-$module)] continue
	if [info exists Notes($module-Color)] {
	    puts $out "set Notes($modVar($filename-$module)-Color) \{$Notes($module-Color)\}"
	}
	if [info exists Notes($module-Position)] {
	    puts $out "set Notes($modVar($filename-$module)-Position) \{$Notes($module-Position)\}"
	}
    }
    
    puts $out "\n\# Create the Connections between Modules"
    # sort by output port # to handle dynamic ports
    set connections [lsort -integer -index 3 [lsort -unique $connections]]
    set i 0
    foreach conn $connections {
	if {![info exists modVar($filename-[oMod conn])] || 
	    ![info exists modVar($filename-[iMod conn])]} continue
	puts -nonewline $out "set c$i \[addConnection "
	puts $out \
	    "$modVar($filename-[oMod conn]) [oNum conn] $modVar($filename-[iMod conn]) [iNum conn]\]"
	incr i
    }
    
    puts $out "\n\# Mark which Connections are Disabled"
    set i 0
    foreach conn $connections {
	if {![info exists modVar($filename-[oMod conn])] || ![info exists modVar($filename-[iMod conn])]} continue
	set id [makeConnID $conn]
	if { [info exists Disabled($id)] && $Disabled($id) } {
	    puts $out "set Disabled(\$c$i) \{1\}"
	}
	incr i
    }
    
    puts $out "\n\# Set the Connection Notes and Dislpay Options"
    set i 0
    foreach conn $connections {
	if {![info exists modVar($filename-[oMod conn])] || ![info exists modVar($filename-[iMod conn])]} continue
	set id [makeConnID $conn]
	if { [info exists Notes($id)] && [string length $Notes($id)] } {
	    puts $out "set Notes(\$c$i) \{$Notes($id)\}"
	}
	if [info exists Notes($id-Color)] {
	    puts $out "set Notes(\$c$i-Color) \{$Notes($id-Color)\}"
	}
	
	if [info exists Notes($id-Position)] {
	    puts $out "set Notes(\$c$i-Position) \{$Notes($id-Position)\}"
	}
	incr i
    }
	
    puts $out "\n\# Set the GUI variables for each Module"
	
    close $out
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
