
catch {rename IsoSurfaceDW ""}

itcl_class PSECommon_Visualization_IsoSurfaceDW {
    inherit Module
    constructor {config} {
	set name IsoSurfaceDW
	set_defaults
    }
    method set_defaults {} {
	global $this-isoval
	set $this-isoval 1
	global $this-emit_surface
	set $this-emit_surface 1
	global $this-auto_update
	set $this-auto_update 0
	global $this-logTCL
	set $this-logTCL 0
	global $this-single
	set $this-single 1
	global $this-method
	set $this-method Hash
	global $this-min $this-max
	set $this-min 0
	set $this-max 200
	global $this-tclBlockSize
	set $this-tclBlockSize 4
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set $this-clr-r 0.5
	set $this-clr-g 0.7
	set $this-clr-b 0.3
    }
    method raiseColor { col } {
	set w .ui[modname]
	if {[winfo exists $w.color]} {
	    raise $w.color
	    return;
	} else {
	    toplevel $w.color
	    global $this-clr
	    makeColorPicker $w.color $this-clr "$this setColor $col" \
		    "destroy $w.color"
	}
    }
    method setColor { col } {
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]

	.ui[modname].f.f.col config -background [format #%04x%04x%04x $ir $ig $ib]
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	global $this-min $this-max
	scale $w.f.isoval -variable $this-isoval \
		-from [set $this-min] -to [set $this-max] -label "IsoValue:" \
		-resolution 0.000001 -showvalue true \
		-orient horizontal \
		-state normal
	pack $w.f.isoval -side top -fill x
	scale $w.f.blocksize -variable $this-tclBlockSize \
		-from 2 -to 64 -label "BlockSize:" \
		-showvalue true -orient horizontal
	pack $w.f.blocksize -side top -fill x

	frame $w.f.b
	frame $w.f.b.l
	checkbutton $w.f.b.l.emit_surface -text "Emit Surface" -relief flat \
		-variable $this-emit_surface
	checkbutton $w.f.b.l.auto_update -text "Auto Update" -relief flat \
		-variable $this-auto_update -command "$this auto"
	global $this-logTCL
	checkbutton $w.f.b.l.log -text "Log Isovalue" -relief flat \
		-variable $this-logTCL -command "$this-c log"
	checkbutton $w.f.b.l.single -text "Single Processor" -relief flat \
		-variable $this-single
	pack $w.f.b.l.emit_surface $w.f.b.l.auto_update $w.f.b.l.log $w.f.b.l.single -side top -expand 1
	make_labeled_radio $w.f.b.r "Method: " "" \
		top $this-method \
		{{Hash "Hash"} {Rings "Rings"} {MC "MC"} {None "None"}}
	pack $w.f.b.l $w.f.b.r -side left -expand 1
	pack $w.f.b -side top -fill x -expand 1
	button $w.f.ex -text "Execute" -command $n
	frame $w.f.f
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]
	frame $w.f.f.col -relief ridge -borderwidth 4 -height 0.7c -width 0.7c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	button $w.f.f.b -text "Set Color" -command "$this raiseColor $w.f.f.col"
	pack $w.f.f.b $w.f.f.col -side left -fill x -padx 5 -expand 1
	pack $w.f.ex $w.f.f -side top -fill x -expand 1
    }
    method set_minmax {min max} {
	global $this-min $this-max
	set $this-min $min
	set $this-max $max
	set w .ui[modname]
	if {[winfo exists $w]} {
	    $w.f.isoval configure -from $min -to $max
	}
    }
    method auto { } {
	set w .ui[modname]
	global $this-auto_update
	if {[set $this-auto_update]} {
	    $w.f.isoval configure -command "$this-c needexecute"
	} else {
	    $w.f.isoval configure -command ""
	}
    }
}
