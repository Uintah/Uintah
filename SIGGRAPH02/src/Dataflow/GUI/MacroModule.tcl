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

# MacroModule.tcl: Merge several modules into one "Macro Module"

set port_spacing 18
set port_width 13
set port_height 7

global iport_mapping
set iport_mapping ""

global MacroedModules
set MacroedModules ""

global groupnums
set groupnums 0

global MModuleFakeConnections


itcl_class macromodule {
    
    method mmodname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
     
    public name
    
    protected mmodframe
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_with 0
    protected mdragged
    protected mconnected
    protected iport_mapping
    protected oport_mapping

    # "Fake" Connections going to an from macromodules
    protected FakeConnections

    
    # Network Connection lines which must be created to accomodate the new
    # MacroModule
    protected MacroModuleConnections

    # Modules contained in macromodule
    protected members
    
    # Network lines contained in macromodule
    protected module_connections

    
    # <Graphical Only> connections to be made when the module is destroyed,
    # or saved...
    protected need_cons


    constructor {} {
	global CurrentlySelectedModules
	global CurrentMacroModules
	global MacroedModules
	global unselected_color


	set MacroModuleConnections ""
	set module_connections ""
	set CurrentMacroModules "$CurrentMacroModules [mmodname]"
	
	set iport_mapping ""
	set oport_mapping ""

	set FakeConnections ""

	set need_cons ""
	global maincanvas minicanvas

	set maincanvas .bot.neteditFrame.canvas
	set minicanvas .top.globalViewFrame.canvas
	
	# Store the modules contained in the MacroModule 
	# in $CurrentlySelectedModules
	
	set members $CurrentlySelectedModules
	set MacroedModules "$MacroedModules $members"
    }

    method get_x {} {
	global maincanvas
	set coords [$maincanvas coords [mmodname]]
	return [lindex $coords 0]
    }

    method get_y {} {
	global maincanvas
	set coords [$maincanvas coords [mmodname]]
	return [lindex $coords 1]
    }

    method append_need_cons { connid } {
	set need_cons "$need_cons $connid"
    }

    method delete_need_cons { conid } {
	set temp_list ""
	
	foreach c $need_cons {
	    if { [string match $conid $c] == 0 } {
		set temp_list "$temp_list $c"
	    }
	}
	
	set need_cons $temp_list
		    }

    method get_need_cons {} {
	return $need_cons
    }

    method get_FakeConnections {} {
	return $FakeConnections
    }
    
    method set_FakeConnections {fc} {
	set FakeConnections $fc
    }

    method get_iport_mapping {} {
	return $iport_mapping
    }
    
    method set_iport_mapping { ipm } {
	set iport_mapping $ipm
    }
	
    method get_oport_mapping {} {
	return $oport_mapping
    }

    method set_oport_mapping { opm } {
	set oport_mapping $opm
    }

    method set_MacroModuleConnections { mmc } {
	set MacroModuleConnections $mmc
    }
    
    method MacroModuleConnections {} {
	return $MacroModuleConnections
    }

    method configureAllIPorts {} {
	global maincanvas
	configureIPorts $maincanvas
    }

    method configureAllOPorts {} {
	global maincanvas
	configureOPorts $maincanvas
    }
   
    method make_icon {canvas minicanvas modx mody} {

	#Make the MacroModule on the maincanvas

	set mmodframe $canvas.macromodule[mmodname]
	frame $mmodframe -relief raised -borderwidth 3

	frame $mmodframe.ff
	pack $mmodframe.ff -side top -expand yes -fill both -padx 5 -pady 6
	
	set p $mmodframe.ff

	global ui_font
	global sci_root
	
	global modname_font
	global time_font

	
	#Make the MacroModule on the minicanvas
	
	global basecolor

	set miniframe $minicanvas.macromodule[mmodname]
	frame $miniframe -borderwidth 0
	frame $miniframe.ff
	pack $miniframe.ff -side top -expand yes \
		-fill both -padx 2 -pady 1
	global SCALEX SCALEY

	$minicanvas create rectangle \
		[expr $modx/$SCALEX] [expr $mody/$SCALEY] \
		[expr $modx/$SCALEX+4] [expr $mody/$SCALEY + 2] \
		-outline "" -fill $basecolor \
		-tags [mmodname]

	# Make the title
	label $p.title -text "[mmodname]" -font $modname_font -anchor w
	pack $p.title -sid top -padx 2 -anchor w

	# Make the individual components here
	

	# Configure Ports
	
	configureIPorts $canvas
	configureOPorts $canvas


	# Stick it in the canvas

	$canvas create window $modx $mody -window $mmodframe \
		-tags [mmodname] -anchor nw
	
	# Try to find a position for the icon where it doesn't
	# overlap other icons
	set done 0
	while { $done == 0 } {
	    set x1 $modx
	    set y1 $mody
	    set x2 [expr $modx+120]
	    set y2 [expr $mody+50]
	    
	    set l [llength [$canvas find overlapping $x1 $y1 $x2 $y2]]
	    

	    if { $l == 0 || $l == 1 || $l == 2} {
		set done 1
	    } else {
		$canvas move [mmodname] 0 80
		$minicanvas move [mmodname] 0 [expr int(80 / $SCALEY)]
		set mody [expr int($mody)]
		incr mody 80
	    }
	}




	# bindings
	bind $p <1> "macro_moduleStartDrag [mmodname] %X %Y"
	bind $p <B1-Motion> "macro_moduleDrag $canvas $minicanvas [mmodname] %X %Y"
	
	bind $p.title <1> "macro_moduleStartDrag [mmodname] %X %Y"
	bind $p.title <B1-Motion> "macro_moduleDrag $canvas $minicanvas [mmodname] %X %Y"

	#Create Menu
	menu $p.menu -tearoff false
		
	$p.menu add command -label "Ungroup..." -command "ungroup_modules \
		$canvas $minicanvas [mmodname]"
	
	bind $p <3> "tk_popup $p.menu %X %Y"
	bind $p.title <3> "tk_popup $p.menu %X %Y"


	#Build Connections
	build_MModuleConnections [mmodname]

    }   

    method mod_type {} {
	return "macromodule"
    }

    method set_members {m} {
	set members $m
    }

    method get_members {} {
	return $members
    }

    method set_connections { connections } {
	set module_connections $connections
    }

    method get_connections {} {
	return $module_connections
    }
    
    method configureIPorts { canvas } {
	global MacroedModules
	
	set i 0
	
	while {[winfo exists $mmodframe.iport$i]} {
	    destroy $mmodframe.iport$i
	    destroy $mmodframe.iportlight$i
	    incr i
	}

	#Configure MacroModule Input Ports
	
	global port_spacing
	global port_width
	global port_height

	set i 0

	# Determine which module's ports to place on the macromodule
	foreach m $members {
	    set iportinfo [$m-c iportinfo]
	    set c 0

	    foreach t $iportinfo {
		set portcolor [lindex $t 0]
		set connected [lindex $t 1]
		set ct 1
		
		#Compute where to place the next dataport
		set x [expr $i*$port_spacing+6]
		
		if {$connected} {
		    set e "outtop"
		    foreach con [netedit getconnected $m] {
			if { ([string match [lindex $con 4] $c] == 1) &&\
				([string match [lindex $con 3] $m] == 1) } {
			    set ct 1
			    foreach mem $members {
				if { [string match $mem [lindex $con 1]] == 1 } { 
				    set ct 0
				}
			    }
			    if {($ct == 1) && ([string match "*$con*" \
				    $MacroModuleConnections] == 0)} {
				set MacroModuleConnections \
					"$MacroModuleConnections {$con}"
			    }
			}
		    }
		} else {
		    set e "top"
		}
		
		if {$ct == 0} {
		    incr c
		    set ct 1
		    continue
		}
		
		bevel $mmodframe.iport$i -width $port_width \
			-height $port_height -borderwidth 3 \
			-edge $e -background $portcolor \
			-pto 2 -pwidth 7 -pborder 2
		place $mmodframe.iport$i -bordermode outside -x $x -y 0 \
			-anchor nw
		frame $mmodframe.iportlight$i -width $port_width -height 4\
			-relief raised -background black -borderwidth 0
		place $mmodframe.iportlight$i -in $mmodframe.iport$i \
			-x 0 -rely 1.0 -anchor nw

		if { [string match "*{$m $c $i}*" "$iport_mapping"] == 0} {
		    set tiport_mapping "$iport_mapping {$m $c $i}"
		    set iport_mapping $tiport_mapping
		}	
		
		bind $mmodframe.iport$i <2> "startIPortConnection\
			[mmodname] $i %x %y"
		bind $mmodframe.iport$i <B2-Motion> \
			"trackIPortConnection [mmodname] $i %x %y"
		bind $mmodframe.iport$i <ButtonRelease-2> \
			"endPortConnection \"$portcolor\""


		incr i
		incr c
	    }
	}
    }


    method configureOPorts { canvas } {
	global MacroedModules
	
	set i 0
	while {[winfo exists $mmodframe.oport$i]} {
	    destroy $mmodframe.oport$i
	    destroy $mmodframe.oportlight$i
	    incr i
	}

	# Configure MacroModule Output Ports

	global port_spacing
	global port_width
	global port_height

	set i 0

	# Determine which of the module's ports to place on the macromodule
	set ct ""

	foreach m $members {
	    set oportinfo [$m-c oportinfo]
	    set c 0

	    foreach t $oportinfo {
		set portcolor [lindex $t 0]
		set connected [lindex $t 1]

		# Compute where to place the next dataport
		set x [expr $i*$port_spacing+6]

		if {$connected} {
		    set e "outbottom"
		    foreach con [netedit getconnected $m] {
			if { ([string match [lindex $con 2] $c] == 1) &&\
				([string match [lindex $con 1] $m] == 1) } {
			    set ct 1
			    foreach mem $members {
				if { [string match $mem [lindex $con 3]] == 1 } {
				    set ct 0
				}
			    }
			    if {($ct == 1) && ([string match "*$con*" \
				    $MacroModuleConnections] == 0)} {
				set MacroModuleConnections \
					"$MacroModuleConnections {$con}"
			    }
			}
		    }
		} else {
		    set e "bottom"
		}

		if {$ct == 0} {
		    incr c
		    set ct 1
		    continue
		}
		
		bevel $mmodframe.oport$i -width $port_width -height \
			$port_height -borderwidth 3 -edge $e -background\
			$portcolor -pto 2 -pwidth 7 -pborder 2
		place $mmodframe.oport$i -bordermode ignore -rely 1 -anchor sw\
			-x $x
		
		frame $mmodframe.oportlight$i -width $port_width -height 4 \
			-relief raised -background black -borderwidth 0
		place $mmodframe.oportlight$i -in $mmodframe.oport$i \
			-x 0 -y 0 -anchor sw

		
		if { [string match "*{$m $c $i}*" "$oport_mapping" ] == 0 } {
		    set toport_mapping "$oport_mapping {$m $c $i}"
		    set oport_mapping $toport_mapping
		    
		}

		bind $mmodframe.oport$i <2> "startOPortConnection\
			[mmodname] $i %x %y"
		bind $mmodframe.oport$i <B2-Motion> \
			"trackOPortConnection [mmodname] $i %x %y"
		bind $mmodframe.oport$i <ButtonRelease-2> \
			"endPortConnection \"$portcolor\""
		incr i
		incr c
	    } 
	}
    }

    method get_real_iport_connections { $port } {
	set conmap [$this mmodule] get_real_connections
    }

    method get_real_oport_connections { $port } {
	
    
    }

}


proc create_mmodule {} {
    set maincanvas .bot.neteditFrame.canvas
    set minicanvas .top.globalViewFrame.canvas
    
    macromodule m
    m make_icon $maincanvas $minicanvas 10 10
}

proc macro_moduleStartDrag {modid x y} {
    global lastX lastY
    set lastX $x
    set lastY $y
    global moduleDragged
    set moduleDragged 0
    global moduleConnected
    #set moduleConnected [netedit getconnected $modid]
    
}

proc macro_moduleDrag {maincanvas minicanvas modid x y} {
    global xminwarped
    global xmaxwarped
    global yminwarped
    global ymaxwarped
    global lastX lastY
    global SCALEX SCALEY
    global CurrentMacroModules

    
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
    set currybdr [expr $canheight + ([lindex [$maincanvas yview] 0]*$mainCanvasHeight)]

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
    ###1##########################################
    
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

    global moduleDragged
    set moduleDragged 1
    global moduleConnected
    rebuildMModuleConnections $modid
}




proc get_iport_num { mmodid modid iwhich } {
    set list [ $mmodid get_iport_mapping ]
    
    foreach l $list {
	if { [string match "$modid" [lindex $l 0]] } {
	    if { [string match "$iwhich" [lindex $l 1]]} {
		return [lindex $l 2]
	    }
	}
    }
    
    #no mapping
    return $iwhich
}

proc get_oport_num {mmodid modid owhich } {
    set list [$mmodid get_oport_mapping ]
    
    foreach l $list {
	if { [string match "$modid" [lindex $l 0]] } {
	    if { [string match "$owhich" [lindex $l 1]]} {
		return [lindex $l 2]
	    }
	}
    }

    #no mapping
    return $owhich

}

proc build_MModuleConnections { mmodid } {
    global maincanvas minicanvas
    global MacroedModules
    global netedit_canvas
    global netedit_mini_canvas

    set bad_connections ""

    # Building MModuleConnections for connections that already contain
    # MModules

    
    foreach m [$mmodid MacroModuleConnections] {
	
	set cid [lindex $m 0]
	set omodid [[lindex $m 1] MacroModule]
	set imodid [[lindex $m 3] MacroModule]
	set color [$maincanvas itemcget [lindex $m 0] -fill]

	set con_source ""

	
	if { [string match $mmodid $omodid] == 1 } {
	    set owhich [get_oport_num $mmodid [lindex $m 1] [lindex $m 2]]
	    set con_source "omodid $omodid"
	    
	} elseif { [string match [$omodid mod_type] "macromodule"] == 1 } {
	    foreach mapping [$omodid get_oport_mapping] {
		set tomodid [lindex $m 1]
		set towhich [lindex $m 2]
		if { [string match [lindex $mapping 0] $tomodid] == 1 } {
		    if { [string match [lindex $mapping 1] $towhich] == 1 } {
			set ow [lindex $mapping 2]
		    }
		}
	    }
	    set owhich $ow
	    set con_source "omodid $omodid"
	} else {
	    set owhich [lindex $m 2]
	}
	
	if { [string match $mmodid $imodid] == 1 } {
	    set iwhich [get_iport_num $mmodid [lindex $m 3] [lindex $m 4]]
	    set con_source "imodid $imodid"
	} elseif { [string match [$imodid mod_type] "macromodule"] == 1 } {
	    foreach mapping [$imodid get_iport_mapping] {
		set timodid [lindex $m 3]
		set tiwhich [lindex $m 4]
		if { [string match [lindex $mapping 0] $timodid] == 1 } {
		    if { [string match [lindex $mapping 1] $tiwhich] == 1} {
			set iw [lindex $mapping 2]
		    }
		}
	    }
	    set iwhich $iw
	} else {
	    set iwhich [lindex $m 4]
	}
	
	set id $omodid
 	append id "_p$owhich"
 	append id "_to_$imodid"
 	append id "_p$iwhich"
	
	if { [string match [$omodid mod_type] "macromodule"] == 1 } {
	    set port "$maincanvas"
	    append port ".macromodule$omodid"
	    append port ".oport$owhich"

	    set color_inf [$port configure -background]
	    set color [lindex $color_inf 4]

	} elseif { [string match [$imodid mod_type] "macromodule"] == 1 } {
	    set port "$maincanvas"
	    append port ".macromodule$imodid"
	    append port ".iport$iwhich"
	    
	    set color_inf [$port configure -background]
	    set color [lindex $color_inf 4]
	}
	

	set mmodfakecon "[lindex $m 0] $color [lindex $m 1] [lindex $m 2]\
		[lindex $m 3] [lindex $m 4] [lindex $m 5] [lindex $m 6]\
		$id $omodid $owhich $imodid $iwhich"
	

	buildConnection $id $color $omodid $owhich $imodid $iwhich

	# When a connection of this type is built connection info for the
	# connection to be deleted from another macromodule must also be
	# included

	global MModuleFakeConnections
	set t 0
	
	

	
	if { [string match [lindex $mmodfakecon 7] $mmodid] == 1 } {
	    #work on input mod
	    
	    set newname "[lindex $mmodfakecon 2]_p"
	    append newname "[lindex $mmodfakecon 3]_to_[lindex $mmodfakecon 9]"
	    append newname "_p[lindex $mmodfakecon 10]"
	    
	    set newcon "[lindex $mmodfakecon 0] [lindex $mmodfakecon 1]\
		    [lindex $mmodfakecon 2] [lindex $mmodfakecon 3]\
		    [lindex $mmodfakecon 4] [lindex $mmodfakecon 5]\
		    $newname\
		    [lindex $mmodfakecon 2] [lindex $mmodfakecon 3]\
		    [lindex $mmodfakecon 9] [lindex $mmodfakecon 10]"
	    
	} elseif { [string match [lindex $mmodfakecon 9] $mmodid] == 1 } {
	    #work on oput mod
	    set newname " [lindex $mmodfakecon 7]_p"
	    append newname "[lindex $mmodfakecon 8]_to_[lindex $mmodfakecon 4]"
	    append newname "_p[lindex $mmodfakecon 5]"

	    set newcon "[lindex $mmodfakecon 0] [lindex $mmodfakecon 1]\
		    [lindex $mmodfakecon 2] [lindex $mmodfakecon 3]\
		    [lindex $mmodfakecon 4] [lindex $mmodfakecon 5]\
		    $newname\
		    [lindex $mmodfakecon 7] [lindex $mmodfakecon 8]\
		    [lindex $mmodfakecon 4] [lindex $mmodfakecon 5]"
	    
	}
	
	set t 0
	foreach mmfc $MModuleFakeConnections {
	    if { [string match $mmfc $newcon] == 1 } {
		set t 1
	    }
	}
	
	if { $t == 0 } {
	    set MModuleFakeConnections "$MModuleFakeConnections\
		    {$newcon}"
	}

	
	set t 0
	foreach mmfc $MModuleFakeConnections {
	    if { [string match $mmfc  $mmodfakecon] == 1 } {
		set t 1
	    }
	}
	
	if { $t == 0 } {
	    set MModuleFakeConnections "$MModuleFakeConnections\
		    {$mmodfakecon}"
	} 
			
	if { [string match "*$id*" [$mmodid get_FakeConnections]] == 0 } {
	    $mmodid set_FakeConnections "[$mmodid get_FakeConnections]\
		    {$id $omodid $owhich $imodid $iwhich}"
	}
	
	
	if { [string match [lindex $con_source 0] "omodid"] == 1} {
	    # Originates from omodid... check imodid
	    if { [string match [$imodid mod_type] "macromodule"] == 1 } {
		foreach fake_con [$imodid get_FakeConnections] {
		    set t 0
		    set real_omodid [lindex [get_real_oport $omodid $owhich] 0]
		    set real_owhich [lindex [get_real_oport $omodid $owhich] 1]
		    		    		    

		    set fake_connid $omodid
		    append fake_connid "_p$owhich"
		    append fake_connid "_to_$imodid"
		    append fake_connid "_p$iwhich"

	
		    
	    		    
		    if { [string match [lindex $fake_con 1] $real_omodid]\
			    == 1 && [string match [lindex $fake_con 2]\
			    $real_owhich] == 1 } {
			if { [string match [lindex $fake_con 3] $imodid] \
				== 1 && [string match [lindex $fake_con 4]\
				$iwhich] == 1} {
			    set old_fake_con $fake_con
			    set t 1
			}
		    }
		    if { $t == 1 } {
			break
		    }
		}
		
		set templist ""
		foreach fake_con [$imodid get_FakeConnections] {
		    if { [string match $fake_con $old_fake_con] == 0 } {
			set templist "$templist {$fake_con}"
		    } else {
			set templist "$templist {$id $omodid $owhich $imodid\
				$iwhich}"
			
			set fake "$id $color $omodid $owhich $imodid\
				$iwhich $fake_con"
									
			# generating fake2...

			# fomodid
			foreach c [ $imodid get_iport_mapping ] {
			    if { [string match [lindex $c 2] $owhich] } {
				set rimodid [lindex $c 0]
				set riwhich [lindex $c 1]
			    }
			}
			
			
			set fconnid $omodid
			append fconnid "_p$owhich"
			append fconnid "_to_$rimodid"
			append fconnid "_p$riwhich"
			
			
			set temp "$fconnid $color $omodid $owhich $rimodid\
				$riwhich $id $omodid $owhich $imodid $iwhich"
			
			set fconnid $temp
						
			set new_fake "[lindex $fake_con 0] $color\
				[lindex $fake_con 1] [lindex $fake_con 2]\
				[lindex $fake_con 3] [lindex $fake_con 4]\
				$id $omodid $owhich $imodid $iwhich"
						

			global MModuleFakeConnections
			
			#uts "Second Case..."
			#uts "new_fake: $new_fake"
						
			set MModuleFakeConnections\
				"$MModuleFakeConnections {$new_fake}\
				{$fconnid}"
			
			$netedit_canvas delete [lindex $fake_con 0]
			$netedit_mini_canvas delete [lindex $fake_con 0]
		    }
		    
		    set bad_connections "$bad_connections $fake_con"
		    
		    $imodid set_FakeConnections $templist
		
		   
		    
		}
	    }
	
	}
	
	if { [string match [lindex $con_source 0] "imodid"] == 1} {
	    # Originates from imodid... check omodid
	    if { [string match [$omodid mod_type] "macromodule"] == 1 } {
		foreach fake_con [$omodid get_FakeConnections] {
		    set t 0
		    set real_imodid [lindex [get_real_iport $imodid $iwhich] 0]
		    set real_iwhich [lindex [get_real_iport $imodid $iwhich] 1]
		    		    
		    if { [string match [lindex $fake_con 1] $omodid]\
			    == 1 && [string match [lindex $fake_con 2]\
			    $owhich] == 1 } {
			
			if { [string match [lindex $fake_con 3] $real_imodid]\
				== 1 && [string match [lindex $fake_con 4]\
				$real_iwhich] == 1 } {		
			    set old_fake_con $fake_con
			    set t 1
			}
		    }
		    if {$t == 1} {
			break
		    }
		}
		set templist ""
		foreach fake_con [$omodid get_FakeConnections] {
		    if { [string match $fake_con $old_fake_con] == 0 } {
			set templist "$templist {$fake_con}"
		    } else {
			
			# generating fake2
						
			foreach m [$omodid get_oport_mapping] {
			    if { [string match [lindex $m 2] $owhich] == 1 } {
				set romodid [lindex $m 0]
				set rowhich [lindex $m 1]
			    }
			}

						
			set fconnid $romodid
			append fconnid "_p$rowhich"
			append fconnid "_to_$imodid"
			append fconnid "_p$iwhich"
			
			set temp "$fconnid $color $romodid $rowhich\
				$imodid $iwhich $id $omodid $owhich $imodid\
				$iwhich"
			
			set fconnid $temp
			
			set fake "$id $color $omodid $owhich $imodid\
				$iwhich $fake_con"

			set new_fake "[lindex $fake_con 0] $color\
				[lindex $fake_con 1] [lindex $fake_con 2]\
				[lindex $fake_con 3] [lindex $fake_con 4]\
				$id $omodid $owhich $imodid $iwhich"
			
		    
			global MModuleFakeConnections
			
			
			set MModuleFakeConnections\
				"$MModuleFakeConnections {$new_fake}\
				{$fconnid}"

			set templist "$templist {$id $omodid $owhich $imodid\
				$iwhich}"
			$netedit_canvas delete [lindex $fake_con 0]
			$netedit_mini_canvas delete [lindex $fake_con 0]
		    }		    
		}
		$omodid set_FakeConnections $templist
		set bad_connections "$bad_connections $fake_con"
	    }
	}
    }
}

proc rebuildMModuleConnections {mmodid} {
    if { $mmodid == "" } {
	return
    }
    
  
    set list [$mmodid get_FakeConnections]
    
    foreach l $list {
	
	set connid [lindex $l 0]
	set omodid [lindex $l 1]
	set owhich [lindex $l 2]
	set imodid [lindex $l 3]
	set iwhich [lindex $l 4]
	
	rebuildConnection $connid $omodid $owhich $imodid $iwhich
    }
}

proc saveMacroModules {} {
    global CurrentMacroModules
    global maincanvas
    foreach mmodule $CurrentMacroModules {
	set g [groupnum]
	foreach member [$mmodule get_members] {
	    global $member-group
	    set $member-group "{{$g} {[$mmodule get_x] [$mmodule get_y]}}"
	    global $member-lastpos
	    set $member-lastpos [$member get_last_pos]
	}
    }
}

proc groupnum {} {
    global groupnums
    set num 1
    while { 1 } {
	if { [string match "*{$num}*" $groupnums] } {
	    incr num
	} else {
	    set groupnums "$groupnums {$num}"
	    return $num
	}
    }
}

proc renameModule {$mmodid} {
    set box .nameWindo
    toplevel $box
    raise $box
    $box configure -title "Rename MacroModule"
    $box configure -background darkgray
}