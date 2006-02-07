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





proc connectionMenu {x y conn cx cy} {
    global Subnet mouseX mouseY
    set mouseX $cx
    set mouseY $cy
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
    global Subnet Disabled ConnectionRoutes
    $menu_id add command -label "Connection" -state disabled
    $menu_id add separator
    $menu_id add command -label "Delete" -command "destroyConnection {$conn} 1"
    if { [insertModuleOnConnectionMenu $conn $menu_id.insertModule] } {
	$menu_id add cascade -label "Insert Module" -menu $menu_id.insertModule
    }
    set connid [makeConnID $conn]
    setIfExists disabled Disabled($connid) 0
    set label [expr $disabled?"Enable":"Disable"]
    $menu_id add command -command "connection$label {$conn}" -label $label

    $menu_id add command -label "Notes" -command "notesWindow $connid"
    if {  [array exists ConnectionRoutes] && \
	      [array names ConnectionRoutes $connid] != "" } {
	$menu_id add command -label "Auto Path" \
	    -command "array unset ConnectionRoutes $connid
                      drawConnections \{\{$conn\}\}"
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
    global Color Disabled ToolTipText Subnet TracedConnections
    foreach conn $connlist {
	if { [llength $conn] != 4 } { 
	    puts "Not drawing invalid connection: $conn" 
	    continue
	}

	set id [makeConnID $conn]
	set path [routeConnection $conn]
	set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
	set minicanvas $Subnet(Subnet$Subnet([oMod conn])_minicanvas)
	setIfExists disabled Disabled($id) 0
	setIfExists color Color($id) red
	set color [expr $disabled?"$Color(ConnDisabled)":"$color"]
	
        if { [info exists TracedConnections] && \
		 ([lsearch $TracedConnections $conn] != -1) } {
	    set color red
	}

	set linetype bline
	set width 7
	if { [envBool SCIRUN_STRAIGHT_CONNECTIONS] } {
	    set linetype line
	    set width 5
	} 
	if { $disabled } {
	    set width 3
	}

	set flags "-width $width -fill \"$color\" -tags $id"
	set miniflags "-width 1 -fill \"$color\" -tags $id"
	if { ![canvasExists $canvas $id] } {
	    eval $canvas create $linetype $path $flags
	    eval $minicanvas create line [scalePath $path] $miniflags

	    $canvas bind $id <1> "$canvas raise $id
                                  traceConnection \{$conn\}
                                  moduleStartDrag [oMod conn] %X %Y 0"
	    $canvas bind $id <B1-Motion> "moduleDrag [oMod conn] %X %Y"
	    $canvas bind $id <ButtonRelease-1> "moduleEndDrag [oMod conn] %X %Y
                                                deleteTraces"
	    $canvas bind $id <Control-Button-1> "$canvas raise $id
						 traceConnection {$conn} 1"
	    $canvas bind $id <Control-Button-2> "destroyConnection {$conn} 1"
	    $canvas bind $id <ButtonPress-2> "setGlobal ClosestSegment -1"
	    $canvas bind $id <B2-Motion> \
		"changeConnectionRoute %x %y \{$conn\}"
	    $canvas bind $id <Shift-Button-2> \
		"splitConnectionRoute %x %y \{$conn\}"
	    $canvas bind $id <ButtonRelease-2> "setGlobal ClosestSegment -1"
#	    $canvas bind $id <Shift-Button-2-Motion> "changeConnectionRoute %x %y \{$conn\}"
	    $canvas bind $id <3> "connectionMenu %X %Y {$conn} %x %y"
	    canvasTooltip $canvas $id $ToolTipText(Connection)
	} else {
	    eval $canvas coords $id $path
	    eval $canvas itemconfigure $id $flags
	    eval $minicanvas coords $id [scalePath $path]
	    eval $minicanvas itemconfigure $miniflags
	}
	$minicanvas lower $id
	drawNotes $id
    }
}

proc getPathLength { path } {
    return [expr [llength $path]/2-1]
}

proc getPathSegment { path seg } {
    if { $seg < [getPathLength $path] } {
	return [lrange $path [expr $seg*2] [expr $seg*2+3]]
    }
}

proc setPathSegment { path segnum seg } {
    if { $segnum >= [getPathLength $path] } return
    return [eval lreplace \{$path\} [expr $segnum*2] [expr $segnum*2+3] $seg]
}

proc setPathSegmentX { path seg newx } {
    set newseg [getPathSegment $path $seg]
    set newseg "$newx [lindex $newseg 1] $newx [lindex $newseg 3]"
    return [setPathSegment $path $seg $newseg]
}

proc setPathSegmentY { path seg newy } {
    set newseg [getPathSegment $path $seg]
    set newseg "[lindex $newseg 0] $newy [lindex $newseg 2] $newy"
    return [setPathSegment $path $seg $newseg]
}

### projectPointLine
# projects point pX, pY along a perpendicular line onto
# the line segment defined by x1,y1 to x2, y2
# returns 1000000,1000000 if the point projects beyond the endpoints
# of the segment
proc projectPointLine { pX pY x1 y1 x2 y2 } {
    set u [computeDist $x1 $y1 $x2 $y2]
    if { $u <= 0.0 } { return "1000000 1000000" }
    set u [expr (($pX-$x1)*($x2-$x1)+($pY-$y1)*($y2-$y1))/($u*$u)]
    if { $u < 0.0 || $u > 1.0 } { return "1000000 1000000" }
    set x [expr $x1+$u*($x2-$x1)]
    set y [expr $y1+$u*($y2-$y1)]
    return "$x $y"
}

### pointLineDist
# The shortest distance from the point (pX,pY) to the line
# segment (x1,y1),(x2,y2)
proc pointLineDist { pX pY x1 y1 x2 y2 } {
    set xy [projectPointLine $pX $pY $x1 $y1 $x2 $y2]
    return [computeDist $pX $pY [lindex $xy 0] [lindex $xy 1]]
}

proc segmentIsHorizontal { seg } {
    return [expr [lindex $seg 1] == [lindex $seg 3]]
}

proc segmentIsVertical { seg } {
    return [expr [lindex $seg 0] == [lindex $seg 2]]
}


proc insertSegmentMidpoint { path segnum } {
    set seg [getPathSegment $path $segnum]
    set mx [expr ([lindex $seg 0]+[lindex $seg 2])/2]
    set my [expr ([lindex $seg 1]+[lindex $seg 3])/2]
    set seg [list \
		 [lindex $seg 0] [lindex $seg 1] \
		 $mx $my $mx $my \
		 [lindex $seg 2] [lindex $seg 3]]
    return [setPathSegment $path $segnum $seg]
}


proc closestConnectionSegment { path mx my } {
    global ClosestSegment
    if { $ClosestSegment == -1 } {
	set shortest 1000000
	set closest -1
	for {set s 0} {$s < [getPathLength $path] } {incr s} {
	    set dist [eval pointLineDist $mx $my [getPathSegment $path $s]]
	    if { $dist < $shortest } {
		set closest $s
		set shortest $dist
	    }
	}
	setGlobal ClosestSegment $closest
    }
    return $ClosestSegment
}

proc splitConnectionRoute { X Y conn } {
    global Subnet
    set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
    set mx [expr $X + [$canvas canvasx 0]]
    set my [expr $Y + [$canvas canvasy 0]]
    set path [routeConnection $conn]
    set closest [closestConnectionSegment $path $mx $my]
    if { $closest == -1 } return
    set path [insertSegmentMidpoint $path $closest]
    setGlobal ConnectionRoutes([makeConnID $conn]) $path
    setGlobal ClosestSegment -1
}


proc changeConnectionRoute { X Y conn } {
    global Subnet ConnectionRoutes
    set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
    set mx [expr $X + [$canvas canvasx 0]]
    set my [expr $Y + [$canvas canvasy 0]]
    set path [routeConnection $conn]
    set closest [closestConnectionSegment $path $mx $my]

    if { $closest <= 0 || $closest >= [getPathLength $path] } return

    set seg [getPathSegment $path $closest]
    if { [lindex $seg 1] == [lindex $seg 3] } {
	set path [setPathSegmentY $path $closest $my]
    } elseif { [lindex $seg 0] == [lindex $seg 2] } {
	set path [setPathSegmentX $path $closest $mx]
    }

    set ConnectionRoutes([makeConnID $conn]) $path
    drawConnections [list $conn]
}

proc traceConnection { conn { traverse 0 } } {
    global Color TracedConnections
    if { [info exists TracedConnections] } return 
    lappend TracedConnections $conn
    unselectAll
    [oMod conn] toggleSelected
    [iMod conn] toggleSelected
    lightPort [oPort conn] $Color(Trace)
    lightPort [iPort conn] $Color(Trace)
    drawConnections [list $conn]
    if { $traverse } { tracePortsBackwards [list [oPort conn] [iPort conn]] }
}


# Deletes red connections on canvas and turns port lights black
proc deleteTraces {} {
    global Subnet Color TracedSubnets TracedConnections
    setIfExists backup TracedConnections ""
    unsetIfExists TracedConnections
    drawConnections $backup
    unselectAll
    lightPort
}

proc disableConnectionID { connid state } {
    global Disabled
    setIfExists disabled Disabled($connid) 0
    # disabledTrace is called when Disabled is written to,
    # it does the actual disabling of the connection in the network
    if { $disabled != $state } {
	set Disabled($connid) $state
    }
}

proc connectionEnable { conn } {
    global Disabled
    set connid [makeConnID $conn]
    setIfExists disabled Disabled($connid) 0
    if { !$disabled } return
    # disabledTrace is called when Disabled is written to,
    # it does the actual disabling of the connection in the network
    set Disabled($connid) 0
}

proc connectionDisable { conn } {
    global Disabled
    set connid [makeConnID $conn]
    setIfExists disabled Disabled($connid) 0
    if { $disabled } return
    # disabledTrace is called when Disabled is written to,
    # it does the actual disabling of the connection in the network
    set Disabled($connid) 1
}




proc drawConnectionTrace { conn } {
    global Subnet Color TracedSubnets
    lappend TracedSubnets $Subnet([oMod conn])
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

#this procedure exists to support SCIRun < v1.20 Networks
proc addConnection { omodid owhich imodid iwhich } {
    update idletasks
    return [createConnection [list $omodid $owhich $imodid $iwhich]]
}

proc createConnection { conn { record_undo 0 } { tell_SCIRun 1 } } {
    global Subnet Notes Disabled Color undoList redoList

    # if the module name is blank discard this call and return with no error
    # this is mainly for loading networks that have unfound modules
    if {![string length [iMod conn]] || ![string length [oMod conn]]} {return}

    # If the in or out module of the connection is exactly "Subnet"
    # append the subnet number of the opposite ends module to the name
    if {[string equal [oMod conn] Subnet]&&[info exists Subnet([iMod conn])]} {
	set conn [lreplace $conn 0 0 Subnet$Subnet([iMod conn])]
    }
    if {[string equal [iMod conn] Subnet]&&[info exists Subnet([oMod conn])]} {
	set conn [lreplace $conn 2 2 Subnet$Subnet([oMod conn])]
    }

    if { [string first Subnet [iMod conn]] != 0 &&
	 [string first Subnet [oMod conn]] != 0 &&
	 ![string equal [lindex [portName [iPort conn]] 0] \
	       [lindex [portName [oPort conn]] 0]] } {
	displayErrorWarningOrInfo "*** Cannot create connection from [oMod conn] output \#[oNum conn] to [iMod conn] input \#[iNum conn].\n*** Port types do not match.\n*** Please fix your network." "error"
	drawConnectionTrace $conn
	return
    }
    
    # make sure the connection lives entirely on the same subnet
    if { ![info exists Subnet([oMod conn])] ||
	 ![info exists Subnet([iMod conn])] ||
	 $Subnet([oMod conn]) != $Subnet([iMod conn]) } {
	puts "Not creating connection $conn: Subnet levels dont match"
	return
    }


    # Trying to create subnet editor interface connection in the main network
    # editor window, happens when user is loading a subnet into the main window
    if {($Subnet([oMod conn]) == 0 && [isaSubnetEditor [oMod conn]]) || \
	($Subnet([iMod conn]) == 0 && [isaSubnetEditor [iMod conn]]) } {
	return
    }
    
    networkHasChanged

    if { $tell_SCIRun} {
	foreach realConn [findRealConnections $conn] {
	    if {[eval netedit addconnection $realConn] == ""} {
		tk_messageBox -type ok -parent . -icon warning -message \
		    "Cannot create connection:\n createConnection $realConn." 
	    }	    
	}
    }
    
    lappend Subnet([oMod conn]_connections) $conn
    if {![string equal [iMod conn] [oMod conn]] } {
	lappend Subnet([iMod conn]_connections) $conn
    }

    set connid [makeConnID $conn]
    unsetIfExists Disabled($connid)
    set Color($connid) [portColor [oPort conn]]
    if ![string length $Color($connid)] { set $Color($connid) red }

    drawConnections [list $conn]
    drawPorts [oMod conn] o
    drawPorts [iMod conn] i
    checkForDisabledModules [oMod conn] [iMod conn]

    if $record_undo {
	set redoList "" ; # new actions invalidate the redo list
	lappend undoList [list "createConnection" $conn]
    }
    return $connid
}
		      


proc destroyConnection { conn { record_undo 0 } { tell_SCIRun 1 } { dont_collapse_dynamic 0 }} { 
    global Subnet Color Disabled Notes undoList redoList
    set connid [makeConnID $conn]

    networkHasChanged
    deleteTraces

    listFindAndRemove Subnet([oMod conn]_connections) $conn
    listFindAndRemove Subnet([iMod conn]_connections) $conn 

    setIfExists disabled Disabled($connid) 0

    array unset Disabled $connid
    array unset Color $connid
    array unset Notes $connid* ;# Delete Notes text, position, & color

    set subnet $Subnet([oMod conn])
    set canvas $Subnet(Subnet${subnet}_canvas)
    set minicanvas $Subnet(Subnet${subnet}_minicanvas)
    $canvas delete $connid $connid-notes $connid-notes-shadow
    $minicanvas delete $connid $connid-notes $connid-notes-shadow

    if { $tell_SCIRun && !$disabled } {
	foreach realConn [findRealConnections $conn] {
	    netedit deleteconnection [makeConnID $realConn] $dont_collapse_dynamic
	}
    }

    if { [isaSubnetEditor [oMod conn]] && 
	 ![llength [portConnections [oPort conn]]] } {
	foreach econn [portConnections "SubnetIcon${subnet} [oNum conn] i"] {
	    destroyConnection $econn
	}
	set iconcanvas $Subnet(Subnet$Subnet(SubnetIcon$subnet)_canvas)
	if { [canvasExists $iconcanvas SubnetIcon${subnet}] } {
	    removePort [oPort conn]
	}
    }

    if { [isaSubnetEditor [iMod conn]] } {
	foreach econn [portConnections "SubnetIcon${subnet} [iNum conn] o"] {
	    destroyConnection $econn
	}
	set iconcanvas $Subnet(Subnet$Subnet(SubnetIcon$subnet)_canvas)
	if { [canvasExists $iconcanvas SubnetIcon${subnet}] } {
	    removePort [iPort conn]
	}
    }
    
    drawPorts [oMod conn] o
    drawPorts [iMod conn] i

    if $record_undo {
	set redoList "" ;# new actions invalidate the redo list
	lappend undoList [list "destroyConnection" $conn]
    } 
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


proc startPortConnection { port } {
    global Subnet modname_font possiblePorts
    unsetIfExists possibleConnection ;# global even though not declared as such
    set subnet $Subnet([pMod port])
    set canvas $Subnet(Subnet${subnet}_canvas)
    $canvas create text [portCoords $port] \
	-text [portName $port] -font $modname_font -tags temp \
	-fill white -anchor [expr [string equal [pType port] o]?"nw":"sw"]
    # if the port is an already connected input port, then stop now
    if { [string equal i [pType port]] && [portIsConnected $port] } { return }

    set possiblePorts [connectablePorts $port]

    # if the subnet is not level 0 (meaning the main network editor)
    # create a subnet input or output port for the module
    if { $subnet } {
	if [string equal i [pType port]] {
	    set dataType [lindex [portName $port] 0]
	    foreach iConn [portConnections "Subnet$subnet all o"] {
		if [string equal $dataType [lindex [portName [iPort iConn]] 0]] {
		    lappend possiblePorts [oPort iConn]
		}
	    }
	}
		
	set addSubnetPort 1
	foreach conn [portConnections $port] {
	    if [string equal Subnet$subnet [[invType port]Mod conn]]  {
		set addSubnetPort 0
		break
	    }
	}	
	if {$addSubnetPort} {
	    set num [portCount "Subnet$subnet 0 [invType port]"]
	    set temp "Subnet$subnet $num [invType port]"
	    drawPort $temp [portColor $port] 0
	    lappend possiblePorts $temp
	}
    }

    foreach poss $possiblePorts {
	set path [routeConnection [makeConn $port $poss]]
	eval $canvas create line $path -width 2 \
	    -tags \"temp tempconnections [join $poss ""]\"
    }
}

proc trackPortConnection { port x y } {
    global possiblePorts possibleConnection Color Subnet
    if { ![info exists possiblePorts] || ![llength $possiblePorts]} return
    set canvas $Subnet(Subnet$Subnet([pMod port])_canvas)
    set portWin $canvas.module[pMod port].port[pType port][pNum port]
    set x [expr $x+[winfo x $portWin]+[lindex [$canvas coords [pMod port]] 0]]
    set y [expr $y+[winfo y $portWin]+[lindex [$canvas coords [pMod port]] 1]]
    set mindist [eval computeDist $x $y [portCoords $port]]
    foreach poss $possiblePorts {
	set dist [eval computeDist $x $y [portCoords $poss]]
	if {$dist < $mindist} {
	    set mindist $dist
	    set possPort $poss
	}
    }
    $canvas itemconfigure tempconnections -fill black
    unsetIfExists possibleConnection ;# global even though not declared as such
    if [info exists possPort] {
	$canvas raise [join $possPort ""]
	$canvas itemconfigure [join $possPort ""] -fill $Color(Trace)
	set possibleConnection [makeConn $possPort $port]
    } 
}

proc endPortConnection { port } {
    global Subnet possibleConnection
    $Subnet(Subnet$Subnet([pMod port])_canvas) delete temp
    if [info exists possibleConnection] {
	createConnection $possibleConnection 1
	unset possibleConnection
    }
    if $Subnet([pMod port]) { drawPorts Subnet$Subnet([pMod port]) }
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
    global undoList redoList Subnet
    if ![llength $undoList] {
	return
    }

    # Get the last action performed
    set undo_item [lindex $undoList end]
    # Remove it from the list
    set undoList [lrange $undoList 0 end-1]
    # Add it to the redo list
    lappend redoList $undo_item

    # if the connection now spans subnets, invalidate undo stack
    set conn [lindex $undo_item 1] 
    if { $Subnet([oMod conn]) != $Subnet([iMod conn]) } {
	set undoList ""
	set redoList ""
	return
    }

    case [lindex $undo_item 0] {
	"createConnection"  { destroyConnection [lindex $undo_item 1] }
	"destroyConnection" { createConnection [lindex $undo_item 1] }
    }
}


proc redo {} {
    global undoList redoList Subnet
    if ![llength $redoList] {
	return
    }
    # Get the last action undone
    set redo_item [lindex $redoList end]
    # Remove it from the list
    listRemove redoList end
    # Add it to the undo list
    lappend undoList $redo_item

    # if the connection now spans subnets, invalidate undo stack
    set conn [lindex $redo_item 1] 
    if { $Subnet([oMod conn]) != $Subnet([iMod conn]) } {
	set undoList ""
	set redoList ""
	return
    }

    eval $redo_item
}

proc routeConnection { conn } {
    set outpos [portCoords [oPort conn]]
    set inpos [portCoords [iPort conn]]
    set ox [expr int([lindex $outpos 0])]
    set oy [expr int([lindex $outpos 1])]
    set ix [expr int([lindex $inpos 0])]
    set iy [expr int([lindex $inpos 1])]

    global ConnectionRoutes
    set connid [makeConnID $conn]
    if {  [array exists ConnectionRoutes] && \
	      [array names ConnectionRoutes $connid] != "" } {
	set path $ConnectionRoutes($connid)
	set path [setPathSegmentX $path 0 $ox]
	set path [lreplace $path 1 1 $oy]
	set last [expr [getPathLength $path]-1]
	set path [setPathSegmentX $path $last $ix]
	set path [lreplace $path end end $iy]
	if { ![string equal $path $ConnectionRoutes($connid)] } {
	    set ConnectionRoutes($connid) $path
	}
	return $path
    }

    if {[envBool SCIRUN_STRAIGHT_CONNECTIONS] } {
	return [list $ox $oy $ox [expr $oy+2] $ix [expr $iy-3] $ix $iy]
    } elseif { $ox == $ix && $oy <= $iy } {
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

proc findModulesToInsertOnConnection { conn } {
    global Subnet ModuleIPorts ModuleOPorts
    set origins [findPortOrigins [oPort conn]]
    if { ![llength $origins] } {
	set origins [findPortOrigins [iPort conn]]
	if { ![llength $origins] } return
    }
    set port [lindex [lsort -integer -unique -index 1 $origins] 0]
    set path [modulePath [pMod port]]
    if { [string equal o [pType port]] } {
	set datatype [lindex $ModuleOPorts($path) [pNum port]]
    } else {
	set datatype [lindex $ModuleIPorts($path) [pNum port]]
    }

    set modules ""
    foreach maybe [array names ModuleIPorts] {
	set onum [lsearch -exact $ModuleOPorts($maybe) $datatype]
	set inum [lsearch -exact $ModuleIPorts($maybe) $datatype]
	if { $onum != -1 && $inum != -1 } {
	    lappend modules "$maybe $onum $inum"
	}
    }
    return [lsort -command moduleCompareCommand $modules]
}


proc insertModuleOnConnectionMenu { conn menu } {
    global ModuleMenu
    # return if there is no information to put in menu
    if { ![info exists ModuleMenu] } { return 0 }
    set moduleList [findModulesToInsertOnConnection $conn]
    if { ![llength $moduleList] } { return 0 }
    # destroy the old menu
    if [winfo exists $menu] {
	return 1
    }

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
	set command "insertModuleOnConnection \{$conn\} $path"
	$submenu add command -label [lindex $path 2] -command $command
    }
    update idletasks
    return 1
}


proc insertModuleOnConnection { conn package category module onum inum } {
    global Subnet mouseX mouseY inserting insertOffset
    set inserting 1
    set insertOffset "0 0"
    set Subnet(Loading) $Subnet([oMod conn])
    if { ![isaSubnetEditor [oMod conn]] } {
	set canvas $Subnet(Subnet$Subnet([oMod conn])_canvas)
	set bbox [$canvas bbox [oMod conn]]
	set mouseX [lindex $bbox 0]
	set mouseY [expr [lindex $bbox 3] + 10]
    }
    set modid [addModuleAtPosition $package $category $module $mouseX $mouseY]
    set Subnet(Loading) 0
    set inserting 0
    destroyConnection $conn 1 1 0
    after 100 createConnection \{[makeConn "$modid $onum o" [iPort conn]]\} 1 1
    after 100 createConnection \{[makeConn [oPort conn] "$modid $inum i"]\} 1 1
}
    



proc computeDist {x1 y1 x2 y2} {
    set dx [expr $x2-$x1]
    set dy [expr $y2-$y1]
    return [expr sqrt($dx*$dx+$dy*$dy)]
}


proc findRealConnections { conn } {
    set connections ""
    foreach iport [findPortOrigins [iPort conn]] {
	foreach oport [findPortOrigins [oPort conn]] {
	    lappend connections [makeConn $oport $iport]
	}
    }
    return [lsort -integer -index 3 $connections]
}


#creates a connection from two ports
proc makeConn { port1 port2 } {
    if [string equal [pType port1] [pType port2]] { return "" }
    if [string equal [pType port1] o] {
	return "[pMod port1] [pNum port1] [pMod port2] [pNum port2]"
    }
    return "[pMod port2] [pNum port2] [pMod port1] [pNum port1]"
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



trace variable ConnectionRoutes wu routeChanged

proc routeChanged { args } {
    set conn [parseConnectionID [lindex $args 1]]
    drawConnections [list $conn]

}