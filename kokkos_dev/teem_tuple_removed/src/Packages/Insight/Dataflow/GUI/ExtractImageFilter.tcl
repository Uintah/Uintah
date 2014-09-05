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
#    File   : ExtractImageFilter.tcl
#    Author : Darby Van Uitert
#    Date   : January 2004

itcl_class Insight_Filters_ExtractImageFilter {
    inherit Module
    constructor {config} {
        set name ExtractImageFilter
        set_defaults
    }

    method set_defaults {} {
	global $this-num-dims
	global $this-reset

	set $this-num-dims 0
	set $this-reset 0
    }

    method set_max_vals {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {

		set ax $this-absmaxDim$i
		set_scale_max_value $w.f.mmf.a$i [set $ax]
		if {[set $this-maxDim$i] == -1 || [set $this-reset]} {
		    set $this-maxDim$i [set $this-absmaxDim$i]
		}
	    }
	    set $this-reset 0
	}
    }

    method init_dims {} {
	for {set i 0} {$i < [set $this-num-dims]} {incr i} {

	    if { [catch { set t [set $this-minDim$i] } ] } {
		set $this-minDim$i 0
	    }
	    if { [catch { set t [set $this-maxDim$i]}] } {
		set $this-maxDim$i -1
	    }
	    if { [catch { set t [set $this-absmaxDim$i]}] } {
		set $this-absmaxDim$i 0
	    }
	}
	make_min_max
    }

    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {
		if {! [winfo exists $w.f.mmf.a$i]} {
		    min_max_widget $w.f.mmf.a$i "Dim $i" \
			$this-minDim$i $this-maxDim$i $this-absmaxDim$i 
		}
	    }
	}
    }

   method clear_dims {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {
		if {[winfo exists $w.f.mmf.a$i]} {
		    destroy $w.f.mmf.a$i
		}
		unset $this-minDim$i
		unset $this-maxDim$i
		unset $this-absmaxDim$i
	    }
	    set $this-reset 1
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.mmf
	pack $w.f.mmf -padx 2 -pady 2 -side top -expand yes
	
	
	if {[set $this-num-dims] == 0} {
	    label $w.f.mmf.t -text "Need to Execute to know the number of Dimensions."
	    pack $w.f.mmf.t
	} else {
	    init_dims 
	}

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}


