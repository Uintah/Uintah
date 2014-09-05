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
	global $this-knownB0
	set $this-knownB0 1

	global $this-use-default-threshold
	set $this-use-default-threshold 1

        global $this-threshold
        set $this-threshold 0.0

        global $this-soft
        set $this-soft 0.0

        global $this-bmatrix
        set $this-bmatrix ""

        global $this-scale
        set $this-scale 0.0


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

	checkbutton $w.f.options.knownB0 -text \
	    "B0 is stored as first DWI value" -variable $this-knownB0
	pack $w.f.options.knownB0 -side top -expand yes -fill x
	checkbutton $w.f.options.usedefaultthreshold -text \
	    "Use Default Threshold" -variable $this-use-default-threshold
	pack $w.f.options.usedefaultthreshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" \
	    -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.soft -labeltext "soft:" \
	    -textvariable $this-soft
        pack $w.f.options.soft -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.scale -labeltext "scale:" \
	    -textvariable $this-scale
        pack $w.f.options.scale -side top -expand yes -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
