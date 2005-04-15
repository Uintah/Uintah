
itcl_class PSECommon_Fields_ShowGeometry {
    inherit Module
    constructor {config} {
	set name ShowGeometry
	set_defaults
    }

    method set_defaults {} {
	global $this-showConP
	global $this-nodeDisplayType
	set $this-nodeDisplayType Spheres
	set $this-showConP 1
	set $this-nodeChan-r 0.5
	set $this-nodeChan-g 0.3
	set $this-nodeChan-b 0.3
	set $this-conChan-r 0.1
	set $this-conChan-g 0.7
	set $this-conChan-b 0.3
	$this-c needexecute
    }

    method raiseColor {col colRoot colMsg} {
	global $colRoot
	set window .ui[modname]
	if {[winfo exists $window.color]} {
	    raise $window.color
	    return;
	} else {
	    toplevel $window.color
	    makeColorPicker $window.color $colRoot \
		    "$this setColor $col $colRoot $colMsg" \
		    "destroy $window.color"
	}
    }

    method setColor {col colRoot colMsg} {
	global $colRoot
	global $colRoot-r
	global $colRoot-g
	global $colRoot-b
	set ir [expr int([set $colRoot-r] * 65535)]
	set ig [expr int([set $colRoot-g] * 65535)]
	set ib [expr int([set $colRoot-b] * 65535)]

	set window .ui[modname]
	$col config -background [format #%04x%04x%04x $ir $ig $ib]
	$this-c $colMsg
    }

    method addColorSelection {frame colRoot colMsg} {
	#add node color picking 
	global $colRoot
	global $colRoot-r
	global $colRoot-g
	global $colRoot-b
	set ir [expr int([set $colRoot-r] * 65535)]
	set ig [expr int([set $colRoot-g] * 65535)]
	set ib [expr int([set $colRoot-b] * 65535)]
	
	frame $frame.colorFrame
	frame $frame.colorFrame.col -relief ridge -borderwidth \
		4 -height 0.7c -width 0.7c \
		-background [format #%04x%04x%04x $ir $ig $ib]
	
	set cmmd "$this raiseColor $frame.colorFrame.col $colRoot $colMsg"
	button $frame.colorFrame.set_color -text "Set Color" -command $cmmd
	
	#pack the node color frame
	pack $frame.colorFrame.set_color $frame.colorFrame.col -side left 
	pack $frame.colorFrame -side bottom 

    }

    method ui {} {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    raise $window
	    return;
	}
	toplevel $window
	
	#frame for all options to live
	frame $window.options

	# node frame holds ui related to vert display (left side)
	frame $window.options.nodeFrame -relief groove -borderwidth 2
	pack $window.options.nodeFrame -padx 2 -pady 2 -side left -fill y
	set n "$this-c needexecute"	

	label $window.options.nodeFrame.frameTitle -text "Node Display Options"

	global $this-nodeDisplayType
	make_labeled_radio $window.options.nodeFrame.radio \
		"Node Display Type" "$this-c nodeSphereP" top \
		$this-nodeDisplayType {Spheres Axes}


	#pack the node radio button
	pack $window.options.nodeFrame.frameTitle \
		$window.options.nodeFrame.radio -side top -fill x

	#add node color picking 
	addColorSelection $window.options.nodeFrame $this-nodeChan \
		nodeColorChange


	# con frame holds ui related to connection display (right side)
	frame $window.options.conFrame -relief groove -borderwidth 2
	pack $window.options.conFrame -padx 2 -pady 2 -side left -fill y

	label $window.options.conFrame.frameTitle \
		-text "Connection Display Options"
	
	checkbutton $window.options.conFrame.showCon \
		-text "Show Connections" \
		-command "$this-c connectionDisplayChange"
	$window.options.conFrame.showCon select

	pack $window.options.conFrame.frameTitle \
		$window.options.conFrame.showCon -side top -fill x

	#add connection color picking 
	addColorSelection $window.options.conFrame $this-conChan \
		conColorChange

	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2
	pack $window.options $window.control -padx 2 -pady 2 -side top

	button $window.control.execute -text Execute -command $n
	button $window.control.dismiss -text Dismiss -command "destroy $window"
	pack $window.control.execute $window.control.dismiss -side left 
    }
}


















