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
	set at [list [$canvas canvasx 0] [$canvas canvasy $border]]
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
    set isoport [string equal [pType port] o]
    set x [expr [pNum port]*$port_spacing+($isSubnetEditor?9:6)]
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
    global Color CurrentlySelectedModules
    lightPort $port $Color(Trace)
    set CurrentlySelectedModules ""
    foreach conn [portConnections $port] {
	lightPort [[invType port]Port conn] $Color(Trace)
	drawConnectionTrace $conn
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
	    #drawConnectionTrace $conn
	    lappend TracedConnections $conn
	    lappend ports [[pType port]Port conn]
	}
    }
}




