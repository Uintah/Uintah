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
	global $this-fixedSizeTCL
	set $this-widgetSizeTCL 1
	set $this-fixedSizeTCL 0
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
	make_entry $w.f.s "WidgetSize:" $this-widgetSizeTCL "$this-c needexecute"
	global $this-fixedSizeTCL
	checkbutton $w.f.f -text "Fixed Size" -variable $this-fixedSizeTCL

	pack $w.f.s $w.f.f -side top -fill x -expand yes
        pack $w.f -side top -fill x -expand yes
    }
}
