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
#    File   : TendSatin.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendSatin ""}

itcl_class Teem_Tend_TendSatin {
    inherit Module
    constructor {config} {
        set name TendSatin
        set_defaults
    }
    method set_defaults {} {
        global anisotropy
        set anisotropy 0.0

        global max
        set max 0.0

        global boundary
        set boundary 0.0

        global thickness
        set thickness 0.0

        global size
        set size 0


    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.anisotropy -labeltext "anisotropy:" -textvariable $this-anisotropy
        pack $w.f.options.anisotropy -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.max -labeltext "max:" -textvariable $this-max
        pack $w.f.options.max -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.boundary -labeltext "boundary:" -textvariable $this-boundary
        pack $w.f.options.boundary -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.thickness -labeltext "thickness:" -textvariable $this-thickness
        pack $w.f.options.thickness -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.size -labeltext "size:" -textvariable $this-size
        pack $w.f.options.size -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
