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
#    File   : TendEstim.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendEstim ""}

itcl_class Teem_Tend_TendEstim {
    inherit Module
    constructor {config} {
        set name TendEstim
        set_defaults
    }
    method set_defaults {} {
        global threshold
        set threshold 0.0

        global soft
        set soft 0.0

        global bmatrix
        set bmatrix ""

        global scale
        set scale 0.0


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

        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.soft -labeltext "soft:" -textvariable $this-soft
        pack $w.f.options.soft -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.bmatrix -labeltext "bmatrix:" -textvariable $this-bmatrix
        pack $w.f.options.bmatrix -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.scale -labeltext "scale:" -textvariable $this-scale
        pack $w.f.options.scale -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
