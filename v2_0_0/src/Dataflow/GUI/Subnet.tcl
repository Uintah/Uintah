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

global Subnet
set Subnet(num) 0



itcl_class SubnetModule {
    inherit Module
    constructor { config } {
	set make_progress_graph 0
	set make_time 0
	set isSubnetModule 1
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

	
proc setupMainSubnet {} {
    global maincanvas Subnet
    set Subnet(num) 0
    set Subnet(Subnet0_canvas) $maincanvas
    set Subnet(Subnet0_Name) "Main"
    foreach modid [getCanvasModules $maincanvas] {
	set Subnet($modid) 0
	lappend Subnet(Subnet0_Modules) $modid
    }
    set Subnet(Subnet0_oportinfo) ""
    set Subnet(Subnet0_iportinfo) ""
    set Subnet(Subnet0_connections) ""
    set Subnet(Subnet0) 0

}

proc numPorts { subnet porttype} {
    set num 0
    global Subnet
    foreach conn $Subnet(Subnet${subnet}_connections) {
	if [string equal [${porttype}Mod conn] Subnet${subnet}] {
	    incr num
	}
    }
    return $num
}

proc updateSubnetName { name1 name2 op } {
    global Subnet
    # extract the Subnet Number from the array index
    set subnet [string range $name2 6 [expr [string first _Name $name2]-1]]
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

proc makeSubnet { from_subnet { x 0 } { y 0 }} {
    global Subnet mainCanvasHeight mainCanvasWidth Color
    incr Subnet(num)
    set Subnet(Subnet$Subnet(num))		$Subnet(num)
    set Subnet(Subnet$Subnet(num)_Name)		"Sub-Network \#$Subnet(num)"
    set Subnet(Subnet$Subnet(num)_Modules)	""
    set Subnet(Subnet$Subnet(num)_oportinfo)	""
    set Subnet(Subnet$Subnet(num)_iportinfo)	""
    set Subnet(Subnet$Subnet(num)_connections)	""

    trace variable Subnet(Subnet$Subnet(num)_Name) w updateSubnetName

    set w .subnet$Subnet(num)
    if {[winfo exists $w]} { destroy $w }
    toplevel $w
    wm withdraw $w
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
    SubnetIcon$Subnet(num) make_icon $x $y
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
    if { $modules == "" } {
	set modules $CurrentlySelectedModules
    }

    # First, delete the connections and module icons from the old canvas
    set connectionList ""
    foreach modid $modules {
	listFindAndRemove Subnet(Subnet${from_subnet}_Modules) $modid
	eval lappend connectionList [getModuleConnections $modid]
    }
    # remove all duplicate connections from the list
    set connectionList [lsort -unique $connectionList]
    # delete connections in decreasing input port number for dynamic ports
    foreach conn [lsort -decreasing -integer -index 3 $connectionList] {
	destroyConnection $conn 0 0
    }    
    
    set bbox "$mouseX $mouseY 0 0"
    if { $modules != "" } {
	set bbox [compute_bbox $Subnet(Subnet${from_subnet}_canvas) $modules]
    }

    set subnet [makeSubnet $from_subnet [lindex $bbox 0] [lindex $bbox 1]]
    
    set Subnet(Subnet${subnet}_Modules) $modules
    # Then move the modules to the new canvas
    foreach modid $modules {
	set canvas $Subnet(Subnet$Subnet($modid)_canvas)
	set modbbox [$canvas bbox $modid]
	$canvas delete $modid
	set x [expr [lindex $modbbox 0] - [lindex $bbox 0] + 10]
	set y [expr [lindex $modbbox 1] - [lindex $bbox 1] + 25]
	set Subnet($modid) $subnet
	$modid make_icon $x $y
    }

    # Create new connections to Subnet Icon and within Subnet Editor
    # Note the 0 0 parameters to createConnection:  SCIRun wont be
    # notified of any creation or deletion of connections between modules
    # since no actual connections are being changed
    foreach conn [sortPorts $subnet $connectionList] {
	if {$Subnet([oMod conn]) == $Subnet([iMod conn])} {
	    createConnection $conn 0 0
	} elseif { $Subnet([oMod conn]) == ${subnet} } {
	    set which [numPorts ${subnet} i]
	    createConnection \
		"[oMod conn] [oNum conn] Subnet${subnet} $which" 0 0
	    createConnection \
		"SubnetIcon${subnet} $which [iMod conn] [iNum conn]" 0 0
	} elseif { $Subnet([iMod conn]) == ${subnet} } {
	    set which [numPorts ${subnet} o]
	    createConnection \
		"Subnet${subnet} $which [iMod conn] [iNum conn]" 0 0
	    createConnection \
		"[oMod conn] [oNum conn] SubnetIcon${subnet} $which" 0 0
	}
    }
    showSubnetWindow $subnet
    unselectAll
}


# Sorts a list of connections by input port position left to right
proc sortPorts { subnet ports } {
    global Subnet
    set xposlist ""
    for {set pnum 0} {$pnum < [llength $ports]} {incr pnum} {
	set port [lindex $ports $pnum]
	if {$Subnet([lindex $port 0]) == $subnet} {
	    set pos [computePortCoords "[lindex $port 0] [lindex $port 1] o"]
	} else {
	    set pos [computePortCoords "[lindex $port 2] [lindex $port 3] i"]
	}
	lappend xposlist [list [lindex $pos 0] $pnum]
    }
    set xposlist [lsort -real -index 0 $xposlist]
    set retval ""
    foreach index $xposlist {
	lappend retval [lindex $ports [lindex $index 1]]
    }
    return $retval
}


proc drawSubnetConnections { subnet } {
    global Subnet
    drawConnections $Subnet(Subnet${subnet}_connections)
}



# This procedure stores away all the global variables that start
# with an "m" to allow nested loads of the networks (see the .net files)
proc backupLoadVars {} {
    global loadVars
    set loadVars(savedNames) [uplevel \#0 info vars m*]
    foreach name $loadVars(savedNames) {
	upvar \#0 $name var
	set loadVars($name) $var
    }
}


proc restoreLoadVars {} {
    global loadVars
    foreach name $loadVars(savedNames) {
	upvar \#0 $name var
	set var $loadVars($name)
    }
}


proc loadSubnet { subnet filename { x 0 } { y 0 } } {
    global Subnet SCIRUN_SRCDIR
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
	    set filename [file join ~ Subnets $netname]
	}
    }

    if { ![file exists $filename] } {
	tk_messageBox -type ok -parent . -icon warning -message \
	    "File \"$filename\" does not exist."
	return
    }

    set subnetNumber [makeSubnet $subnet $x $y]
    set oldLoadingLevel $Subnet(Loading)
    set Subnet(Loading) $subnetNumber
    backupLoadVars
    uplevel \#0 source \{$filename\}
    restoreLoadVars
    set Subnet(Subnet$Subnet(Loading)_filename) "$filename"
    set Subnet(Loading) $oldLoadingLevel
    return SubnetIcon$subnetNumber
}


proc saveSubnet { subnet { name ""} } {
    global Subnet SCIRUN_SRCDIR
    if { $name == "" } {
	set name [join [file split $Subnet(Subnet${subnet}_Name) "/"] ""].net
	set home [file dirname ~]
	if [file writable $SCIRUN_SRCDIR] {
	    set dir $SCIRUN_SRCDIR/Subnets
	    file mkdir $dir
	} elseif [file writable $home] {
	    set dir $home/Subnets
	    file mkdir $dir
	} else {
	    tk_messageBox -type ok -parent . -icon error -message \
		"Cannot save $name to $SCIRUN_SRCDIR/Subnets or $home/Subnets" 
		return
	}
	set name $dir/$name
	set Subnet(Subnet${subnet}_filename) $name
    }

    set out [open $name w]
    puts $out "\# SCIRun 2.0 Sub-Network\n"
    puts $out "::netedit dontschedule\n"
    puts -nonewline $out "set Subnet(Subnet\$Subnet(Loading)_Name) \{"
    puts $out "$Subnet(Subnet${subnet}_Name)\}"
    set i 0
    set connections ""
    set m(Subnet${subnet}) "Subnet\$Subnet(Loading)"
    foreach module $Subnet(Subnet${subnet}_Modules) {
	puts -nonewline $out "set m${i} \["
	if { [isaSubnetIcon $module] } {
	    set iconsubnet $Subnet(${module}_num)
	    puts -nonewline $out "loadSubnet \$Subnet(Loading) "
	    if ![info exists Subnet(Subnet${iconsubnet}_filename)] {
		saveSubnet $iconsubnet
	    }
	    set splitname [file split $Subnet(Subnet${iconsubnet}_filename)] 
	    puts -nonewline $out "\"[lindex $splitname end]\" "
	} else {
	    puts -nonewline $out  "addModuleAtPosition "
	    foreach name [lrange [split $module _] 0 2] {
		puts -nonewline $out  "\"$name\" "
	    }
	}
	puts $out "[expr int([$module get_x])] [expr int([$module get_y])]\]"
	eval lappend connections $Subnet(${module}_connections)
	set m($module) "\$m$i"
	incr i
    }
    puts $out ""    
    # sort by output port # to handle dynamic ports
    foreach conn [lsort -integer -index 3 [lsort -unique $connections]] {
	puts -nonewline $out "addConnection "
	puts $out \
	    "$m([oMod conn]) [oNum conn] $m([iMod conn]) [iNum conn]"
    }
    set invalidvars [list -msgStream -done_bld_icon]
    puts $out ""
    set i 0
    foreach module $Subnet(Subnet${subnet}_Modules) {
	foreach var [uplevel \#0 info vars "$module-*"] {
	    set varname [string range $var [string length $module] end]
	    if {[lsearch $invalidvars $varname] == -1} {
		set value [uplevel \#0 set $var]
		puts $out "set $m($module)${varname} \{$value\}"
	    }
	}
    }
    puts $out "\n::netedit scheduleok"
    close $out
    return $name
}		          
    
proc showSubnetWindow { subnet { bbox "" } } {
    wm deiconify .subnet${subnet}
    raise .subnet${subnet}
    global Subnet
    if { $bbox == "" } {
	if { [info exists Subnet(Subnet${subnet}_Modules) ] } {
	    set bbox [compute_bbox $Subnet(Subnet${subnet}_canvas) \
			  $Subnet(Subnet${subnet}_Modules)]
	} else {
	    set bbox { 0 0 0 0 }
	}	
    }
    set minx 90
    set miny 200	
    set wid [expr [lindex $bbox 2]+$minx]
    set hei [expr [lindex $bbox 3]+$miny]
    set maxsize [wm maxsize .]
    set wid [expr $wid>[lindex $maxsize 0]?[lindex $maxsize 0]:$wid]
    set hei [expr $hei>[lindex $maxsize 1]?[lindex $maxsize 1]:$hei]
    set wid [expr $wid<$minx?$minx:$wid]
    set hei [expr $hei<$miny?$miny:$hei]
    wm geometry .subnet${subnet} ${wid}x${hei}
}
