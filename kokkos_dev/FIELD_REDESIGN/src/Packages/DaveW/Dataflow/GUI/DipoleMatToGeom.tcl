##
 #  DipoleMatToGeom.tcl: Set theta and phi for the dipole
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename DaveW_FEM_DipoleMatToGeom ""}

itcl_class DaveW_FEM_DipoleMatToGeom {
    inherit Module
    constructor {config} {
        set name DipoleMatToGeom
        set_defaults
    }
    method set_defaults {} {
	global $this-widgetSizeTCL
	global $this-scaleModeTCL
	global $this-showLastVecTCL
	global $this-showLinesTCL
	set $this-widgetSizeTCL 1
	set $this-scaleModeTCL normalize
	set $this-showLastVecTCL 0
	set $this-showLinesTCL 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
	global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui$[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
        frame $w.f
	global $this-widgetSizeTCL
	make_entry $w.f.s "Widget Size:" $this-widgetSizeTCL "$this-c needexecute"
	frame $w.f.r -relief sunken -bd 2
	global $this-scaleModeTCL
	radiobutton $w.f.r.fixed -text "Fixed Size" -value "fixed" -variable $this-scaleModeTCL
	radiobutton $w.f.r.normalize -text "Normalize Largest" -value "normalize" -variable $this-scaleModeTCL
	radiobutton $w.f.r.scale -text "Scale Size" -value "scale" -variable $this-scaleModeTCL
	pack $w.f.r.fixed $w.f.r.normalize $w.f.r.scale -side top -fill both -expand yes
	global $this-showLastVecTCL
	checkbutton $w.f.v -text "Show Last As Vector" -variable $this-showLastVecTCL
	global $this-showLinesTCL
	checkbutton $w.f.l -text "Show Lines" -variable $this-showLinesTCL

	pack $w.f.s $w.f.r $w.f.v $w.f.l -side top -fill x -expand yes
        pack $w.f -side top -fill x -expand yes
    }
}
