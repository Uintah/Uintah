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
#    File   : Unu3op.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Unu_Unu3op ""}

itcl_class Teem_Unu_Unu3op {
    inherit Module
    constructor {config} {
        set name Unu3op
        set_defaults
    }
    method set_defaults {} {
        global $this-operator
        set $this-operator "+"

	global $this-float1
	set $this-float1 "0.0"

	global $this-float2
	set $this-float2 "0.0"

	global $this-float3
	set $this-float3 "0.0"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.operator -labeltext "Operator:" -textvariable $this-operator
        pack $w.f.options.operator -side top -expand yes -fill x
	
	iwidgets::entryfield $w.f.options.float1 \
	    -labeltext "Float Input 1:" -textvariable $this-float1
        pack $w.f.options.float1 -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.float2 \
	    -labeltext "Float Input 2:" -textvariable $this-float2
        pack $w.f.options.float2 -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.float3 \
	    -labeltext "Float Input 3:" -textvariable $this-float3
        pack $w.f.options.float3 -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
