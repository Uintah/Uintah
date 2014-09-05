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

##
 #  SetupFEMatrix.tcl
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996, March 2001
 #  Copyright (C) 1996 SCI Group
 ##

catch {rename BioPSE_Forward_SetupFEMatrix ""}

itcl_class BioPSE_Forward_SetupFEMatrix {
    inherit Module
    constructor {config} {
        set name SetupFEMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-BCFlag
	global $this-UseCondTCL
        set $this-BCFlag "none"
	set $this-UseCondTCL 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-BCFlag
        make_labeled_radio $w.f.r "Boundary Conditions:" "" \
                left $this-BCFlag \
                {{"Apply Dirichlet" DirSub} \
                {"Ground Node Zero" GroundZero}
                {"No Dirichlet" none}}
	
	global $this-UseCondTCL
	checkbutton $w.f.b -text "Use Conductivities" -variable $this-UseCondTCL -onvalue 1 -offvalue 0
	pack $w.f.r $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
