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

itcl_class BioPSE_Modeling_ModifyConductivities {
    inherit Module
    constructor {config} {
        set name ModifyConductivities
        set_defaults
    }


    method set_defaults {} {
	global $this-num-entries


	set $this-num-entries 0
    }

    method create_entries {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    # Create the new variables and entries if needed.
	    for {set i 0} {$i < [set $this-num-entries]} {incr i} {
		
		if { [catch { set t [set $this-names-$i] } ] } {
		    set $this-names-$i default-$i
		}
		if { [catch { set t [set $this-sizes-$i]}] } {
		    set $this-sizes-$i 1.0
		}
		if { [catch { set t [set $this-m00-$i]}] } {
		    set $this-m00-$i 1.0
		}
		if { [catch { set t [set $this-m01-$i]}] } {
		    set $this-m01-$i 0.0
		}
		if { [catch { set t [set $this-m02-$i]}] } {
		    set $this-m02-$i 0.0
		}
		if { [catch { set t [set $this-m10-$i]}] } {
		    set $this-m10-$i 0.0
		}
		if { [catch { set t [set $this-m11-$i]}] } {
		    set $this-m11-$i 1.0
		}
		if { [catch { set t [set $this-m12-$i]}] } {
		    set $this-m12-$i 0.0
		}
		if { [catch { set t [set $this-m20-$i]}] } {
		    set $this-m20-$i 0.0
		}
		if { [catch { set t [set $this-m21-$i]}] } {
		    set $this-m21-$i 0.0
		}
		if { [catch { set t [set $this-m22-$i]}] } {
		    set $this-m22-$i 1.0
		}

		if {![winfo exists $w.f.tensors.e-$i]} {
		    frame $w.f.tensors.e-$i
		    entry $w.f.tensors.e-$i.name \
			-textvariable $this-names-$i -width 16
		    entry $w.f.tensors.e-$i.scale \
			-textvariable $this-sizes-$i -width 8
		    entry $w.f.tensors.e-$i.m00 \
			-textvariable $this-m00-$i -width 6
		    entry $w.f.tensors.e-$i.m01 \
			-textvariable $this-m01-$i -width 6
		    entry $w.f.tensors.e-$i.m02 \
			-textvariable $this-m02-$i -width 6
		    entry $w.f.tensors.e-$i.m10 \
			-textvariable $this-m10-$i -width 6
		    entry $w.f.tensors.e-$i.m11 \
			-textvariable $this-m11-$i -width 6
		    entry $w.f.tensors.e-$i.m12 \
			-textvariable $this-m12-$i -width 6
		    entry $w.f.tensors.e-$i.m20 \
			-textvariable $this-m20-$i -width 6
		    entry $w.f.tensors.e-$i.m21 \
			-textvariable $this-m21-$i -width 6
		    entry $w.f.tensors.e-$i.m22 \
			-textvariable $this-m22-$i -width 6
		    pack $w.f.tensors.e-$i.name $w.f.tensors.e-$i.scale \
			$w.f.tensors.e-$i.m00 \
			$w.f.tensors.e-$i.m01 \
			$w.f.tensors.e-$i.m02 \
			$w.f.tensors.e-$i.m10 \
			$w.f.tensors.e-$i.m11 \
			$w.f.tensors.e-$i.m12 \
			$w.f.tensors.e-$i.m20 \
			$w.f.tensors.e-$i.m21 \
			$w.f.tensors.e-$i.m22 \
			-side left
		    pack $w.f.tensors.e-$i
		}
	    }

	    # Destroy all the left over entries from prior runs.
	    while {[winfo exists $w.f.tensors.e-$i]} {
		destroy $w.f.tensors.e-$i
		incr i
	    }
	}
    }

    method ui {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	wm minsize $w 200 150

	frame $w.f
	
	frame $w.f.tensors

	scrollbar $w.f.tensors.v -orient vertical -command "$w.f.tensors yview"
	pack $w.f.tensors.v -side right -expand y -fill y

	frame $w.f.tensors.title
	label $w.f.tensors.title.name -text "Material Name" \
	    -width 16 -relief groove
	label $w.f.tensors.title.scale -text "Scale" -width 8 -relief groove
	label $w.f.tensors.title.m00 -text "M00" -width 6 -relief groove
	label $w.f.tensors.title.m01 -text "M01" -width 6 -relief groove
	label $w.f.tensors.title.m02 -text "M02" -width 6 -relief groove
	label $w.f.tensors.title.m10 -text "M10" -width 6 -relief groove
	label $w.f.tensors.title.m11 -text "M11" -width 6 -relief groove
	label $w.f.tensors.title.m12 -text "M12" -width 6 -relief groove
	label $w.f.tensors.title.m20 -text "M20" -width 6 -relief groove
	label $w.f.tensors.title.m21 -text "M21" -width 6 -relief groove
	label $w.f.tensors.title.m22 -text "M22" -width 6 -relief groove
	pack $w.f.tensors.title.name $w.f.tensors.title.scale \
	    $w.f.tensors.title.m00 \
	    $w.f.tensors.title.m01 \
	    $w.f.tensors.title.m02 \
	    $w.f.tensors.title.m10 \
	    $w.f.tensors.title.m11 \
	    $w.f.tensors.title.m12 \
	    $w.f.tensors.title.m20 \
	    $w.f.tensors.title.m21 \
	    $w.f.tensors.title.m22 \
	    -side left
	pack $w.f.tensors.title -side top

	frame $w.f.controls
	button $w.f.controls.execute -text "Execute" \
	    -command "$this-c needexecute"
	button $w.f.controls.reset -text "Reset" \
	    -command "$this-c reset_gui"
	pack $w.f.controls.execute $w.f.controls.reset \
	    -side left -fill x -expand y

	pack $w.f.tensors -side top -anchor n
	pack $w.f.controls -anchor s -expand y -fill x
	pack $w.f

	create_entries
    }
}


