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


catch {rename ChangeCoordinates ""}

itcl_class SCIRun_FieldsGeometry_ChangeCoordinates {
    inherit Module
    
    constructor {config} {
	set name ChangeCoordinates
	set_defaults
    }
    
    method set_defaults {} {
	global $this-oldsystem
	global $this-newsystem
	set $this-oldsystem "Cartesian"
	set $this-newsystem "Spherical"
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	frame $w.f.old
	frame $w.f.new
	
	label $w.f.old.l -text "Input coordinate system: "
	radiobutton $w.f.old.e -text "Cartesian   " -variable $this-oldsystem \
	    -value "Cartesian"
	radiobutton $w.f.old.s -text "Spherical   " -variable $this-oldsystem \
	    -value "Spherical"
	radiobutton $w.f.old.p -text "Polar   " -variable $this-oldsystem \
	    -value "Polar"
	radiobutton $w.f.old.r -text "Range" -variable $this-oldsystem \
	    -value "Range"
	pack $w.f.old.l $w.f.old.e $w.f.old.s $w.f.old.p $w.f.old.r -side left

	label $w.f.new.l -text "Output coordinate system: "
	radiobutton $w.f.new.e -text "Cartesian   " -variable $this-newsystem \
	    -value "Cartesian"
	radiobutton $w.f.new.s -text "Spherical   " -variable $this-newsystem \
	    -value "Spherical"
	radiobutton $w.f.new.p -text "Polar   " -variable $this-newsystem \
	    -value "Polar"
	radiobutton $w.f.new.r -text "Range" -variable $this-newsystem \
	    -value "Range"
	pack $w.f.new.l $w.f.new.e $w.f.new.s $w.f.new.p $w.f.new.r -side left

	button $w.f.e -text "Execute" -command "$this-c needexecute"
	pack $w.f.old $w.f.new $w.f.e -side top
    }
}
