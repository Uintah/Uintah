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
#    File   : UnuCrop.tcl
#    Author : Martin Cole
#    Date   : Tue Mar 18 08:46:53 2003

catch {rename Teem_Unu_UnuCrop ""}

itcl_class Teem_Unu_UnuCrop {
    inherit Module
    constructor {config} {
        set name UnuCrop
        set_defaults
    }
    method set_defaults {} {
	global $this-num-axes
	global $this-reset

	set $this-num-axes 0
	set $this-reset 0
    }

    method set_max_vals {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {

		set ax $this-absmaxAxis$i
		set_scale_max_value $w.f.mmf.a$i [set $ax]
		if {[set $this-maxAxis$i] == -1 || [set $this-reset]} {
		    set $this-maxAxis$i [set $this-absmaxAxis$i]
		}
	    }
	    set $this-reset 0
	}
    }

    method init_axes {} {
	for {set i 0} {$i < [set $this-num-axes]} {incr i} {
	    #puts "init_axes----$i"

	    if { [catch { set t [set $this-minAxis$i] } ] } {
		set $this-minAxis$i 0
		#puts "made minAxis$i"
	    }
	    if { [catch { set t [set $this-maxAxis$i]}] } {
		set $this-maxAxis$i -1
		#puts "made maxAxis$i   [set $this-maxAxis$i]"
	    }
	    if { [catch { set t [set $this-absmaxAxis$i]}] } {
		set $this-absmaxAxis$i 0
		#puts "made absmaxAxis$i"
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
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.mmf.a$i]} {
		    min_max_widget $w.f.mmf.a$i "Axis $i" \
			$this-minAxis$i $this-maxAxis$i $this-absmaxAxis$i 
		}
	    }
	}
    }

   method clear_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {
		#puts $i
		if {[winfo exists $w.f.mmf.a$i]} {
		    destroy $w.f.mmf.a$i
		}
		unset $this-minAxis$i
		unset $this-maxAxis$i
		unset $this-absmaxAxis$i
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
	
	if {[set $this-num-axes] == 0} {
	    label $w.f.mmf.t -text "Need Execute to know the number of Axes."
	    pack $w.f.mmf.t
	} else {
	    init_axes 
	}

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
