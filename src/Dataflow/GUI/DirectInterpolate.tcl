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

itcl_class SCIRun_FieldsData_DirectInterpolate {
    inherit Module
    constructor {config} {
        set name DirectInterpolate
        set_defaults
    }

    method set_defaults {} {
	global $this-interpolation_basis
	global $this-map_source_to_single_dest
	global $this-exhaustive_search
	global $this-exhaustive_search_max_dist
	global $this-np
	set $this-interpolation_basis linear
	set $this-map_source_to_single_dest 0
	set $this-exhaustive_search 1
	set $this-exhaustive_search_max_dist -1
	set $this-np 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.basis
	label $w.basis.label -text "Interpolation Basis:"
	radiobutton $w.basis.const -text "Constant ('find closest')" \
		-variable $this-interpolation_basis -value constant
	frame $w.basis.cframe 
	label $w.basis.cframe.label -text "Constant Mapping:"
	radiobutton $w.basis.cframe.onetomany -text \
		"Each destination gets nearest source value" \
		-variable $this-map_source_to_single_dest -value 0
	radiobutton $w.basis.cframe.onetoone -text \
		"Each source projects to just one destination" \
		-variable $this-map_source_to_single_dest -value 1
	pack $w.basis.cframe.label -side top -anchor w
	pack $w.basis.cframe.onetomany $w.basis.cframe.onetoone \
		-side top -anchor w -padx 15
	radiobutton $w.basis.lin -text "Linear (`weighted')" \
		-variable $this-interpolation_basis -value linear
	pack $w.basis.label -side top -anchor w
	pack $w.basis.const -padx 15 -side top -anchor w
	pack $w.basis.cframe -padx 30 -side top -anchor w
	pack $w.basis.lin -padx 15 -side top -anchor w
	
	frame $w.exhaustive
	label $w.exhaustive.label -text "Exhaustive Search Options:"
	checkbutton $w.exhaustive.check \
	    -text "Use Exhaustive Search if Fast Search Fails" \
	    -variable $this-exhaustive_search
	frame $w.exhaustive.dist
	label $w.exhaustive.dist.label -text \
		"Maximum Distance (negative value -> 'no max'):"
	entry $w.exhaustive.dist.entry -textvariable \
	    $this-exhaustive_search_max_dist -width 8
	pack $w.exhaustive.dist.label $w.exhaustive.dist.entry \
	    -side left -anchor n
	pack $w.exhaustive.label -side top -anchor w
	pack $w.exhaustive.check -side top -anchor w -padx 15
	pack $w.exhaustive.dist -side top -anchor w -padx 30

	scale $w.scale -orient horizontal -variable $this-np -from 1 -to 32 \
		-showvalue true -label "Number of Threads"
	
	frame $w.buttons
	button $w.buttons.execute -text "Execute" \
	    -command "$this-c needexecute"
	button $w.buttons.close -text "Close" -command "destroy $w"
	pack $w.buttons.execute $w.buttons.close -side left -padx 40

	pack $w.basis -side top -anchor w
	pack $w.exhaustive -side top -anchor w -pady 15
	pack $w.scale -side top -expand 1 -fill x
	pack $w.buttons -side top -pady 5
    }
}
