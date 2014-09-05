
#
#  GenSurface.tcl
#
#  Written by:
#   Steven Parker
#   Department of Computer Science
#   University of Utah
#   June 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Surface_GenSurface {
    inherit Module
    constructor {config} {
	set name GenSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-surfacetype
	set $this-surfacetype point
	global $this-cyl_p1-x $this-cyl_p1-y $this-cyl_p1-z
	set $this-cyl_p1-x 0
	set $this-cyl_p1-y 0
	set $this-cyl_p1-z 0
	global $this-cyl_p2-x $this-cyl_p2-y $this-cyl_p2-z
	set $this-cyl_p1-x 0
	set $this-cyl_p1-y 0
	set $this-cyl_p1-z 1
	global $this-cyl_rad
	set $this-cyl_rad .1
	global $this-cyl_nu $this-cyl_nv $this-cyl_ndiscu
	set $this-cyl_nu 4
	set $this-cyl_nv 30
	set $this-cyl_ndiscu 4

	global $this-sph_cen-x $this-sph_cen-y $this-sph_cen-z
	set $this-sph_cen-x 0
	set $this-sph_cen-y 0
	set $this-sph_cen-z 0
	global $this-sph_axis-x $this-sph_axis-y $this-sph_axis-z
	set $this-sph_axis-x 0
	set $this-sph_axis-y 0
	set $this-sph_axis-z 0
	global $this-sph_rad
	set $this-sph_rad .1
	global $this-sph_nu $this-sph_nv
	set $this-sph_nu 20
	set $this-sph_nv 30

	global $this-point_pos-x $this-point_pos-y $this-point_pos-z
	set $this-point_pos-x 0
	set $this-point_pos-y 0
	set $this-point_pos-z 0
	global $this-point_rad
	set $this-point_rad .1

	global $this-cyl_boundary_expr
	set $this-cyl_boundary_expr ""
	global $this-sph_boundary_expr
	set $this-sph_boundary_expr ""
	global $this-point_boundary_expr
	set $this-point_boundary_expr ""
	$this-c needexecute

	global $this-oldst
	set $this-oldst cylinder
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
	global $this-cyl_rad
	puts "rad is [set $this-cyl_rad]";
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 20
	set n "$this-c needexecute "

	make_labeled_radio $w.type "Surface Type:" "$this switch_type yes" \
		top $this-surfacetype \
		{{Cylinder cylinder} {Sphere sphere} {Point point}}
	pack $w.type -side top -padx 5 -anchor w -side top

	frame $w.cylinder -borderwidth 2 -relief groove
	label $w.cylinder.cyl -text "Cylinder:"
	pack $w.cylinder.cyl -side top -pady 5 -anchor w
	make_entry $w.cylinder.bc "Voltage:" \
		$this-cyl_boundary_expr $n
	puts "rad is [set $this-cyl_rad]";
	make_entry $w.cylinder.rad "Radius:" $this-cyl_rad $n
	puts "rad is [set $this-cyl_rad]";
	make_entry $w.cylinder.nu "nu:" $this-cyl_nu $n
	make_entry $w.cylinder.nv "nv:" $this-cyl_nv $n
	make_entry $w.cylinder.ndiscu "ndiscu:" $this-cyl_ndiscu $n
	pack $w.cylinder.bc $w.cylinder.rad $w.cylinder.nu $w.cylinder.nv \
		$w.cylinder.ndiscu -fill x

	frame $w.sphere -borderwidth 2 -relief groove
	label $w.sphere.lab -text "Sphere:"
	pack $w.sphere.lab -side top -pady 5 -anchor w
	make_entry $w.sphere.bc "Voltage:" $this-sph_boundary_expr $n
	make_entry $w.sphere.rad "Radius:" $this-sph_rad $n
	make_entry $w.sphere.nu "nu:" $this-sph_nu $n
	make_entry $w.sphere.nv "nv:" $this-sph_nv $n
	pack $w.sphere.bc $w.sphere.rad $w.sphere.nu $w.sphere.nv -fill x

	frame $w.point -borderwidth 2 -relief groove
	label $w.point.lab -text "Point:"
	pack $w.point.lab -side top -pady 5 -anchor w
	make_entry $w.point.bc "Value:" $this-point_val $n
	pack $w.point.bc -fill x

	switch_type no
    }
    method switch_type {exec} {
	set w .ui[modname]
	global $this-oldst $this-surfacetype
	pack forget $w.[set $this-oldst]
	pack $w.[set $this-surfacetype] \
		-side top -padx 2 -pady 2 -ipadx 2 -ipady 2 -fill x
	set $this-oldst [set $this-surfacetype]
	if {$exec} {
	    $this-c needexecute
	}
    }
}
