##
 #  Coregister.tcl: The coregistration UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #
 #  Copyright (C) 1996 SCI Group
 # 
 #  Log Information:
 #
 #  $Log$
 #  Revision 1.1  1999/08/24 06:22:55  dmw
 #  Added in everything for the DaveW branch
 #
 #  Revision 1.1.1.1  1999/04/24 23:12:17  dav
 #  Import sources
 #
 #  Revision 1.12  1999/01/04 05:31:55  dmw
 #  See Dave for details...
 #
 #  Revision 1.11  1997/03/11 22:05:53  dweinste
 #  fixed stuff
 #
 #  Revision 1.10  1996/10/22 20:00:05  dweinste
 #  nothing.
 #
 #  Revision 1.9  1996/10/22 19:57:52  dweinste
 #  it finally works!!
 #
 #  Revision 1.8  1996/10/22 00:12:19  dweinste
 #  still trying to get this to work!!!!
 #
 #  Revision 1.7  1996/10/21 23:51:54  dweinste
 #  weirdness
 #
 #  Revision 1.6  1996/10/21 23:07:55  dweinste
 #  ugh.
 #
 #  Revision 1.5  1996/10/21 23:03:04  dweinste
 #  seeing if log info works
 #
 ##

itcl_class Coregister {
    inherit Module
    constructor {config} {
        set name Coregister
        set_defaults
    }
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    method set_defaults {} {
        global $this-rot_r_x
        global $this-rot_r_y
        global $this-rot_r_z
        global $this-rot_d_x
        global $this-rot_d_y
        global $this-rot_d_z
        global $this-rot_i_x
        global $this-rot_i_y
        global $this-rot_i_z
        global $this-trans_x
        global $this-trans_y
        global $this-trans_z
	global $this-reg_error
	global $this-iters
	global $this-curr_iter
	global $this-abortButton
	global $this-percent
	global $this-scale
	global $this-transform
	global $this-fiducial
	global $this-fiducialMethod
	global $this-useFirstSurfPts
        set $this-rot_r_x 1
        set $this-rot_r_y 0
        set $this-rot_r_z 0
        set $this-rot_d_x 0
        set $this-rot_d_y 1 
        set $this-rot_d_z 0
        set $this-rot_i_x 0
        set $this-rot_i_y 0
        set $this-rot_i_z 1
        set $this-trans_x 0
        set $this-trans_y 0 
        set $this-trans_z 0
	set $this-reg_error 0
	set $this-curr_iter 0
	set $this-iters 26
	set $this-abortButton 0
	set $this-percent 10
	set $this-scale 1.0
	set $this-transform translate
	set $this-fiducial none
	set $this-fiducialMethod LSF
	set $this-useFirstSurfPts 0
	set $this-useScale 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 450 120
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        set n "$this-c needexecute "
        global $this-rot_r_x
        global $this-rot_r_y
        global $this-rot_r_z
        global $this-rot_d_x
        global $this-rot_d_y
        global $this-rot_d_z
        global $this-rot_i_x
        global $this-rot_i_y
        global $this-rot_i_z
        global $this-trans_x
        global $this-trans_y
        global $this-trans_z
	global $this-reg_error
	global $this-transform
        frame $w.f.v -relief raised -borderwidth 2
	frame $w.f.v.r -relief groove -borderwidth 2
	label $w.f.v.r.l -text "Rotation Vectors"
	frame $w.f.v.r.r
	label $w.f.v.r.r.l1 -text "X: ("
	label $w.f.v.r.r.x -textvariable $this-rot_r_x
	label $w.f.v.r.r.l2 -text ", "
	label $w.f.v.r.r.y -textvariable $this-rot_r_y
	label $w.f.v.r.r.l3 -text ", "
	label $w.f.v.r.r.z -textvariable $this-rot_r_z
	label $w.f.v.r.r.l4 -text ")"
	pack $w.f.v.r.r.l1 $w.f.v.r.r.x $w.f.v.r.r.l2 $w.f.v.r.r.y $w.f.v.r.r.l3 $w.f.v.r.r.z $w.f.v.r.r.l4 -side left -expand 1 -fill x
	frame $w.f.v.r.d
	label $w.f.v.r.d.l1 -text "Y: ("
	label $w.f.v.r.d.x -textvariable $this-rot_d_x
	label $w.f.v.r.d.l2 -text ", "
	label $w.f.v.r.d.y -textvariable $this-rot_d_y
	label $w.f.v.r.d.l3 -text ", "
	label $w.f.v.r.d.z -textvariable $this-rot_d_z
	label $w.f.v.r.d.l4 -text ")"
	pack $w.f.v.r.d.l1 $w.f.v.r.d.x $w.f.v.r.d.l2 $w.f.v.r.d.y $w.f.v.r.d.l3 $w.f.v.r.d.z $w.f.v.r.d.l4 -side left -expand 1 -fill x
	frame $w.f.v.r.i
	label $w.f.v.r.i.l1 -text "Z: ("
	label $w.f.v.r.i.x -textvariable $this-rot_i_x
	label $w.f.v.r.i.l2 -text ", "
	label $w.f.v.r.i.y -textvariable $this-rot_i_y
	label $w.f.v.r.i.l3 -text ", "
	label $w.f.v.r.i.z -textvariable $this-rot_i_z
	label $w.f.v.r.i.l4 -text ")"
	pack $w.f.v.r.i.l1 $w.f.v.r.i.x $w.f.v.r.i.l2 $w.f.v.r.i.y $w.f.v.r.i.l3 $w.f.v.r.i.z $w.f.v.r.i.l4 -side left -expand 1 -fill x
	pack $w.f.v.r.l $w.f.v.r.r $w.f.v.r.d $w.f.v.r.i -side top -expand 1 -fill x
	frame $w.f.v.t -relief groove -width 110 -borderwidth 2
	label $w.f.v.t.l -text "Translate"
	frame $w.f.v.t.x
	label $w.f.v.t.x.l -text "X: "
	label $w.f.v.t.x.x -textvariable $this-trans_x
	pack $w.f.v.t.x.l $w.f.v.t.x.x -side left
	frame $w.f.v.t.y
	label $w.f.v.t.y.l -text "Y: "
	label $w.f.v.t.y.x -textvariable $this-trans_y
	pack $w.f.v.t.y.l $w.f.v.t.y.x -side left
	frame $w.f.v.t.z
	label $w.f.v.t.z.l -text "Z: "
	label $w.f.v.t.z.x -textvariable $this-trans_z
	pack $w.f.v.t.z.l $w.f.v.t.z.x -side left
	pack $w.f.v.t.l -side top -fill x -expand 1
	pack $w.f.v.t.x $w.f.v.t.y $w.f.v.t.z -side top
	pack $w.f.v.r $w.f.v.t -side left -fill x -expand 1
	make_labeled_radio $w.f.tr "Transformation: " "" \
		left $this-transform \
		{{"Both" all} \
		{"Rotate" rotate} \
		{"Translate" translate}}
	global $this-fiducial
	make_labeled_radio $w.f.fiducial "Fiducial Selection Point: " "$this fid" \
		left $this-fiducial \
		{{"None" none} \
		{"Nasion" nasion}
		{"Left" left} \
		{"Right" right}}	
	global $this-useFirstSurfPts
	checkbutton $w.f.cb -text "Use first three surface pts as fiducials?"\
		-variable $this-useFirstSurfPts -command "$this-c firstthree"
	frame $w.f.e
	label $w.f.e.l -text "RMS Error (cm): "
	label $w.f.e.e -textvariable $this-reg_error
	pack $w.f.e.l $w.f.e.e -side left -expand 1
	frame $w.f.fr
	global $this-fiducialMethod
	make_labeled_radio $w.f.fr.fidfit "Fiducial Registration: " " " \
		left $this-fiducialMethod \
		{{"Least Squares Fit" LSF} \
		{"Maxillary Axis" maxAxis}}
	frame $w.f.sc
	checkbutton $w.f.sc.b -text "Use scale?" -variable $this-useScale
	global $this-scale
	label $w.f.sc.l -textvariable $this-scale
	pack $w.f.sc.b $w.f.sc.l -side left
	button $w.f.fr.b -text "Go!" -command "$this-c fiducialFit"
	pack $w.f.fr.fidfit $w.f.fr.b -side left -padx 4
	frame $w.f.b
	label $w.f.b.l1 -text "Auto Registration:   "
	label $w.f.b.l2 -text "Iter#"
	global $this-curr_iter
	label $w.f.b.l3 -width 3 -textvariable $this-curr_iter
	button $w.f.b.auto -text "Go!" -command "$this-c auto"
	global $this-abortButton
	checkbutton $w.f.b.abort -text "Abort" -variable $this-abortButton
	pack $w.f.b.l1 $w.f.b.auto $w.f.b.l2 $w.f.b.l3 $w.f.b.abort -side left -expand 1 -padx 10
	frame $w.f.p
	label $w.f.p.l -text "Percent of Nodes to use for auto registration: "
	entry $w.f.p.e -relief sunken -width 3 -textvariable $this-percent
	pack $w.f.p.l $w.f.p.e -side left -expand 1
        frame $w.f.ti
	frame $w.f.ti.i
	label $w.f.ti.i.l -text "Max Iterations: "
	entry $w.f.ti.i.e -relief sunken -width 4 -textvariable $this-iters
	pack $w.f.ti.i.l $w.f.ti.i.e -side left -expand 1
        pack $w.f.ti.i -padx 15 -side left
	frame $w.f.mgd
	pack $w.f.e $w.f.fr $w.f.fiducial $w.f.cb $w.f.b $w.f.p $w.f.tr \
		$w.f.ti $w.f.sc -side top
#       pack $w.f.v -side bottom -fill x -expand 1
	frame $w.f.pr
	button $w.f.pr.print -text "Print Trans" -command "$this printit"
	button $w.f.pr.print2 -text "Print Pts" -command "$this-c print"
	pack $w.f.pr.print $w.f.pr.print2 -side left -padx 8
	pack $w.f.pr -side bottom
	pack $w.f -expand 1 -fill both
    }
    method fid {} {
	global $this-fiducial
#	puts [set $this-fiducial]
	$this-c [set $this-fiducial]
    }
    method printit {} {
        global $this-rot_r_x
        global $this-rot_r_y
        global $this-rot_r_z
        global $this-rot_d_x
        global $this-rot_d_y
        global $this-rot_d_z
        global $this-rot_i_x
        global $this-rot_i_y
        global $this-rot_i_z
        global $this-trans_x
        global $this-trans_y
        global $this-trans_z
	puts -nonewline "Xrot  "
	puts -nonewline [set $this-rot_r_x]
	puts -nonewline ", "
	puts -nonewline [set $this-rot_r_y]
	puts -nonewline ", "
	puts [set $this-rot_r_z]
	puts -nonewline "Yrot  "
	puts -nonewline [set $this-rot_d_x]
	puts -nonewline ", "
	puts -nonewline [set $this-rot_d_y]
	puts -nonewline ", "
	puts [set $this-rot_d_z]
	puts -nonewline "Zrot  "
	puts -nonewline [set $this-rot_i_x]
	puts -nonewline ", "
	puts -nonewline [set $this-rot_i_y]
	puts -nonewline ", "
	puts [set $this-rot_i_z]
	puts -nonewline "Trans "
	puts -nonewline [set $this-trans_x]
	puts -nonewline ", "
	puts -nonewline [set $this-trans_y]
	puts -nonewline ", "
	puts [set $this-trans_z]
    }
}
