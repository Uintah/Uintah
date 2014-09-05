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


proc connectablePorts { port } { 
    if { [string equal i [pType port]] && [portIsConnected $port] } { return }
    global Subnet
    set ports ""
    set thisDataType [lindex [portName $port] 0]
    foreach modid $Subnet(Subnet$Subnet([pMod port])_Modules) {
	set nPorts [portCount "$modid 0 [invType port]"]
	for {set i 0} {$i < $nPorts} {incr i} {
	    set theirPort "$modid $i [invType port]"
	    if [string equal $thisDataType [lindex [portName $theirPort] 0]] {
		if { [string equal o [pType theirPort]] ||
		     ![portIsConnected $theirPort] } {
		    lappend ports $theirPort
		}
	    }
	}
    }
    return $ports
}

proc findPortOrigins { port } {
    global Subnet
    set ports ""
    lappend portsTodo $port
    while { [llength $portsTodo] } {
	set port [lindex $portsTodo end]
	set portsTodo [lrange $portsTodo 0 end-1]
	if ![isaSubnet [pMod port]] {
	    lappend ports $port
	} else {
	    if { [isaSubnetIcon [pMod port]] } {
		set mod Subnet$Subnet([pMod port]_num)
	    } else {
		set mod SubnetIcon$Subnet([pMod port])
	    }
	    foreach conn [portConnections "$mod [pNum port] [invType port]"] {
		lappend portsTodo [[pType port]Port conn]
	    }
	}
    }
    return $ports
}


proc portCoords { port } {
    global Subnet port_spacing port_width
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set isoport [string equal o [pType port]]
    if { [isaSubnetEditor [pMod port]] } {
	set border [expr $isoport?0:[winfo height $canvas]]
	set at [list [expr [$canvas canvasx 0]+4] [$canvas canvasy $border]]
	set h 0
    } elseif { [lsearch $Subnet(Subnet$Subnet([pMod port])_Modules) [pMod port]]!= -1} {
	set at [$canvas coords [pMod port]]
	set h [winfo height $canvas.module[pMod port]]
    } else {
	return [list 0 0]
    }
    
    set x [expr [pNum port]*$port_spacing+6+$port_width/2+[lindex $at 0]]
    set y [expr ($isoport?$h:0) + [lindex $at 1]]
    return [list $x $y]
}

proc portColor { port } {
    global Subnet
    if { [isaSubnetEditor [pMod port]] } {
	set port "SubnetIcon$Subnet([pMod port]) [pNum port] [invType port]"
    }
    set port [lindex [findPortOrigins $port] 0]
    if ![string length $port] { return red }
    return [[pMod port]-c [pType port]portcolor [pNum port]]
}


proc portName { port } {
    global Subnet
    if { [isaSubnetEditor [pMod port]] } {
	set port "SubnetIcon$Subnet([pMod port]) [pNum port] [invType port]"
    }
    set port [lindex [findPortOrigins $port] 0]
    if ![string length $port] { return "None None" }
    return [[pMod port]-c [pType port]portname [pNum port]]
}

proc portIsConnected { port } {
    global Subnet
    if [info exists Subnet([pMod port]_connections)] {
	foreach conn $Subnet([pMod port]_connections) {
	    if { [pNum port] == [[pType port]Num conn] &&
		 [string equal [[pType port]Mod conn] [pMod port]] } {
		return 1
	    }
	}
    }
    return 0
}

# returns the # of ports of the same type (input or output) on the module
# note: the port # (second item in list) is not used in this procedure
proc portCount { port } {    
    global Subnet
    if [isaSubnetIcon [pMod port]] {
	set port "Subnet$Subnet([pMod port]_num) 0 [invType port]"
    }
    if [isaSubnetEditor [pMod port]] {
	set idx [expr [string equal o [pType port]]?1:3]
	set conns [portConnections "[pMod port] all [pType port]"]
	set conns [lsort -integer -decreasing -index $idx $conns]
	if ![llength $conns] { return 0 }
	return [expr [lindex [lindex $conns 0] $idx]+1]
    } else {
	return [[pMod port]-c [pType port]portcount]
    }
}


proc portConnections { port } {
    global Subnet
    set connections ""
    if [string equal all [pNum port]] { set all 1 } else { set all 0 }
    if [info exists Subnet([pMod port]_connections)] {
	foreach conn $Subnet([pMod port]_connections) {
	    if { ($all || [pNum port] == [[pType port]Num conn]) &&
		 [string equal [pMod port] [[pType port]Mod conn]] } {
		lappend connections $conn
	    }
	}
    }
    return $connections
}



proc drawPorts { modid { porttypes "i o" } } {
    global Subnet
    if { ![info exists Subnet($modid)] } { return }
    set subnet $Subnet($modid)
    if [isaSubnetEditor $modid] {
	drawPorts SubnetIcon$subnet
	set modframe .subnet${subnet}.can
    } else {
	set modframe $Subnet(Subnet${subnet}_canvas).module$modid
	$modid resize_icon
    }
    foreach porttype $porttypes {
	set i 0
	while {[winfo exists $modframe.port$porttype$i]} {
	    destroy $modframe.port$porttype$i
	    destroy $modframe.portlight$porttype$i
	    incr i
	}
	set num [portCount "$modid 0 $porttype"]
	for {set i 0 } {$i < $num} {incr i} {
	    set port [list $modid $i $porttype]
	    drawPort $port [portColor $port] [portIsConnected $port]
	}
    }
}

proc drawPort { port { color red } { connected 0 } } {
    global Subnet ToolTipText
    global port_spacing port_width port_height port_light_height
    set isSubnetEditor [isaSubnetEditor [pMod port]]
    set subnet $Subnet([pMod port])
    if $isSubnetEditor {
	set modframe .subnet${subnet}.can
    } else {
	set modframe $Subnet(Subnet${subnet}_canvas).module[pMod port]
    }
    if { ![winfo exists $modframe] } return
    set isoport [string equal [pType port] o]
    set x [expr [pNum port]*$port_spacing+($isSubnetEditor?13:6)]
    set e [expr $connected?"out":""][expr $isoport?"bottom":"top"]
    set portbevel $modframe.port[pType port][pNum port]
    set portlight $modframe.portlight[pType port][pNum port]
    bevel $portbevel -width $port_width -height $port_height \
	-borderwidth 3 -edge $e -background $color \
	-pto 2 -pwidth 7 -pborder 2
    frame $portlight -width $port_width -height 4 \
	-relief raised -background black -borderwidth 0

    Tooltip $portlight $ToolTipText(ModulePortlight)
    Tooltip $portbevel $ToolTipText(ModulePort)

    if { $isSubnetEditor && $isoport } {
	place $portbevel -bordermode outside \
	    -y $port_light_height -anchor nw -x $x
    } elseif { $isSubnetEditor && !$isoport } {
	place $portbevel -bordermode ignore -x $x -rely 1 -y -4 -anchor sw
    } elseif { !$isSubnetEditor && $isoport } {
	place $portbevel -bordermode ignore -rely 1 -anchor sw -x $x
    } elseif { !$isSubnetEditor && !$isoport } {
	place $portbevel -bordermode outside -x $x -y 0 -anchor nw
    }

    if $isoport {
	place $portlight -in $portbevel -x 0 -y 0 -anchor sw
    } else {
	place $portlight -in $portbevel -x 0 -rely 1.0 -anchor nw
    }
    if !$isSubnetEditor {
	foreach p [list $portbevel $portlight] {
	    bind $p <2> "startPortConnection {$port}"
	    bind $p <B2-Motion> "trackPortConnection {$port} %x %y"
	    bind $p <ButtonRelease-2> "endPortConnection {$port}"
	    bind $p <ButtonPress-1> "tracePort {$port}"
	    bind $p <Control-Button-1> "tracePort {$port} 1"
	    bind $p <ButtonRelease-1> "deleteTraces"
	    bind $p <3> "portMenu %X %Y {$port} %x %y"
	}
    }
}



proc lightPort { { port "" } { color "black" } } {
    global Subnet LitPorts

    if ![string length $port] {
 	if [info exists LitPorts] {
 	    foreach port $LitPorts {
 		lightPort $port black
 	    }
	    set LitPorts [lreplace $LitPorts 0 end]
 	}
 	return
    }

    if {![info exists Subnet([pMod port])]} return

    if {![info exists LitPorts] || [lsearch $LitPorts $port] == -1} { 
	lappend LitPorts $port
    }
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set p $canvas.module[pMod port].portlight[pType port][pNum port]
    if {[winfo exists $p]} {
  	$p configure -background $color
    }
}

#todo Undo stuff
proc removePort { port } {
    global Subnet Color Disabled Notes
    set numPorts [portCount $port]
    for {set i [expr [pNum port]+1]} {$i < $numPorts} {incr i} {
	if [isaSubnetEditor [pMod port]] {
	    set icon SubnetIcon$Subnet([pMod port])
	    foreach conn $Subnet(${icon}_connections) {
		if { [[invType port]Num conn] == $i && \
			 [string equal $icon [[invType port]Mod conn]] } {
		    listFindAndRemove Subnet([oMod conn]_connections) $conn
		    listFindAndRemove Subnet([iMod conn]_connections) $conn
		    set connid [makeConnID $conn]
		    $Subnet(Subnet$Subnet([oMod conn])_canvas) delete \
			$connid $connid-notes $connid-notes-shadow
		    $Subnet(Subnet$Subnet([oMod conn])_minicanvas) delete \
			$connid $connid-notes $connid-notes-shadow
		    set num [expr $i-1]
		    set thisport [list $icon $num [invType port]]
		    set otherport [[pType port]Port conn]
		    set newconn [makeConn $otherport $thisport]
		    lappend Subnet([oMod newconn]_connections) $newconn
		    lappend Subnet([iMod newconn]_connections) $newconn
		    set newconnid [makeConnID $newconn]
		    renameGlobal Notes($newconnid) Notes($connid)
		    renameGlobal Notes($newconnid-Position) Notes($connid-Position)
		    renameGlobal Notes($newconnid-Color) Notes($connid-Color)
		    renameGlobal Color($newconnid) Color($connid)    
		    renameGlobal Disabled($newconnid) Disabled($connid)
		    
		    drawConnections [list $newconn]
		}
	    }
	}

	foreach conn $Subnet([pMod port]_connections) {
	    if { [[pType port]Num conn] == $i &&
		 [string equal [[pType port]Mod conn] [pMod port]] } {
		listFindAndRemove Subnet([oMod conn]_connections) $conn
		listFindAndRemove Subnet([iMod conn]_connections) $conn
		set connid [makeConnID $conn]
		$Subnet(Subnet$Subnet([oMod conn])_canvas) delete \
		    $connid $connid-notes $connid-notes-shadow
		$Subnet(Subnet$Subnet([oMod conn])_minicanvas) delete \
		    $connid $connid-notes $connid-notes-shadow
		set num [expr [[pType port]Num conn]-1]
		set thisport [list [pMod port] $num [pType port]]
		set otherport [[invType port]Port conn]
		set newconn [makeConn $otherport $thisport]
		lappend Subnet([oMod newconn]_connections) $newconn
		lappend Subnet([iMod newconn]_connections) $newconn
		set newconnid [makeConnID $newconn]
		renameGlobal Notes($newconnid) Notes($connid)
		renameGlobal Notes($newconnid-Position) Notes($connid-Position)
		renameGlobal Notes($newconnid-Color) Notes($connid-Color)
		renameGlobal Color($newconnid) Color($connid)    
		renameGlobal Disabled($newconnid) Disabled($connid)

		drawConnections [list $newconn]
	    }
	}
    }
}

proc tracePort { port { traverse 0 }} {
    global Color CurrentlySelectedModules TracedConnections
    lightPort $port $Color(Trace)
    set CurrentlySelectedModules ""
    foreach conn [portConnections $port] {
	lightPort [[invType port]Port conn] $Color(Trace)
	lappend TracedConnections $conn
	drawConnections [list $conn]
	if $traverse { tracePortsBackwards [list [[invType port]Port conn]] }
    }
}

proc tracePortsBackwards { ports } {
    global Subnet TracedConnections
    set backwardTracedPorts ""
    set subnet $Subnet([lindex [lindex $ports 0] 0])
    while { [llength $ports] } {
	set port [lindex $ports end]
	set ports [lrange $ports 0 end-1]
	if { ![isaSubnetEditor [pMod port]] && \
		 $Subnet([pMod port]) == $subnet && \
		 ![[pMod port] is_selected] } {
	    [pMod port] toggleSelected
	}
	if { [lsearch $backwardTracedPorts $port] != -1 } { continue }
	lappend backwardTracedPorts $port
	if [isaSubnetIcon [pMod port]] {
	    set do "Subnet$Subnet([pMod port]_num) [pNum port] [invType port]"
	} elseif [isaSubnetEditor [pMod port]] {
	    set do "SubnetIcon$Subnet([pMod port]) [pNum port] [invType port]"
	} else {
	    set do "[pMod port] all [invType port]"
	}
	foreach conn [portConnections $do] { 
	    lappend TracedConnections $conn
	    drawConnections [list $conn]
	    lappend ports [[pType port]Port conn]
	}
    }
}


proc findModulesToInsertOnPort { port } {
    global Subnet ModuleIPorts ModuleOPorts
    set origins [findPortOrigins $port]
    if { ![llength $origins] } { return }
    set port [lindex [lsort -integer -unique -index 1 $origins] 0]
    set path [modulePath [pMod port]]
    if { [pType port] == "o" } {
	set datatype [lindex $ModuleOPorts($path) [pNum port]]
    } else {
	set lastport [expr [llength $ModuleIPorts($path)]-1]
	# Assume this port was dynamically created
	if { $lastport < [pNum port] } {
	    set port [lreplace $port 1 1 $lastport]
	}
	set datatype [lindex $ModuleIPorts($path) [pNum port]]
    }

    set modules ""
    foreach maybe [array names ModuleIPorts] {
	if { [pType port] == "o" } {
	    set num [lsearch -exact $ModuleIPorts($maybe) $datatype]
	} else {
	    set num [lsearch -exact $ModuleOPorts($maybe) $datatype]
	}
	if { $num != -1 } {
	    lappend modules "$maybe $num"
	}
    }
    return [lsort -command moduleCompareCommand $modules]
}


proc insertModuleOnPortMenu { port menu } {
    # Return if this menu already exists
    if { [winfo exists $menu] } { return 1 }
    # Return if there is no modules that would insert on this port
    set moduleList [findModulesToInsertOnPort $port]
    if { ![llength $moduleList] } { return 0 }

    # create a new menu
    menu $menu -tearoff false -disabledforeground black

    set added ""
    foreach path $moduleList {
	if { [lsearch $added [lindex $path 0]] == -1 } {
	    lappend added [lindex $path 0]
	    # Add a menu separator if this package isn't the first one
	    if { [$menu index end] != "none" } {
		$menu add separator 
	    }
	    # Add a label for the Package name
	    $menu add command -label [lindex $path 0] -state disabled
	}

	set submenu $menu.menu_[join [lrange $path 0 1] _]
	if { ![winfo exists $submenu] } {
	    menu $submenu -tearoff false
	    $menu add cascade -label "  [lindex $path 1]" -menu $submenu
	}
	set command "insertModuleOnPort \{$port\} $path"
	$submenu add command -label [lindex $path 2] -command $command
    }
    update idletasks
    return 1
}


proc insertModuleOnPort { port package category module num } {
    global Subnet mouseX mouseY inserting insertOffset
    set inserting 0
    set insertOffset "0 0"
    set Subnet(Loading) $Subnet([pMod port])
    if { ![isaSubnetEditor [pMod port]] } {
	set c1 [portCoords $port]
	set c2 [portCoords [lreplace $port 1 1 $num]]
	set dx [expr [lindex $c1 0] - [lindex $c2 0]]
	set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
	set bbox [$canvas bbox [pMod port]]
	set mouseX [expr $dx+[lindex $bbox 0] - [$canvas canvasx 0]]
	set y0 [$canvas canvasy 0]
	if { [pType port] == "o" } {
	    set mouseY [expr [lindex $bbox 3] + 20 - $y0]
	} else {
	    set mouseY [expr 2*[lindex $bbox 1] - [lindex $bbox 3] - 25 - $y0]
	}
    }
    set modid [addModuleAtPosition $package $category $module $mouseX $mouseY]
    set Subnet(Loading) 0
    set inserting 0
    after 100 createConnection \{[makeConn $port "$modid $num [invType port]"]\} 1 1
}
    

proc portMenu {x y port cx cy} {
    if { [pType port] == "i" && [portIsConnected $port] } return
    global Subnet mouseX mouseY
    set mouseX $cx
    set mouseY $cy
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set menu_id "$canvas.menu[join $port _]"
    if { [insertModuleOnPortMenu $port $menu_id] } {
	tk_popup $menu_id $x $y
    }
}

