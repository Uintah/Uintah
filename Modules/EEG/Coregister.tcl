itcl_class Coregister {
    inherit Module
    constructor {config} {
        set name Coregister
        set_defaults
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
	global $this-error_metric
	global $this-tolerance
	global $this-iters
	global $this-curr_iter
	global $this-abortButton
	global $this-newtonB
	global $this-percent
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
	set $this-error_metric dist2
	set $this-tolerance 0.0001
	set $this-curr_iter 0
	set $this-iters 50
	set $this-abortButton 0
	set $this-newtonB 1
	set $this-percent 10
    }
    method ui {} {
        set w .ui$this
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 350 200
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
        frame $w.f.v
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
	pack $w.f.v.r.r.l1 $w.f.v.r.r.x $w.f.v.r.r.l2 $w.f.v.r.r.y $w.f.v.r.r.l3 $w.f.v.r.r.z $w.f.v.r.r.l4 -side left
	frame $w.f.v.r.d
	label $w.f.v.r.d.l1 -text "Y: ("
	label $w.f.v.r.d.x -textvariable $this-rot_d_x
	label $w.f.v.r.d.l2 -text ", "
	label $w.f.v.r.d.y -textvariable $this-rot_d_y
	label $w.f.v.r.d.l3 -text ", "
	label $w.f.v.r.d.z -textvariable $this-rot_d_z
	label $w.f.v.r.d.l4 -text ")"
	pack $w.f.v.r.d.l1 $w.f.v.r.d.x $w.f.v.r.d.l2 $w.f.v.r.d.y $w.f.v.r.d.l3 $w.f.v.r.d.z $w.f.v.r.d.l4 -side left
	frame $w.f.v.r.i
	label $w.f.v.r.i.l1 -text "Z: ("
	label $w.f.v.r.i.x -textvariable $this-rot_i_x
	label $w.f.v.r.i.l2 -text ", "
	label $w.f.v.r.i.y -textvariable $this-rot_i_y
	label $w.f.v.r.i.l3 -text ", "
	label $w.f.v.r.i.z -textvariable $this-rot_i_z
	label $w.f.v.r.i.l4 -text ")"
	pack $w.f.v.r.i.l1 $w.f.v.r.i.x $w.f.v.r.i.l2 $w.f.v.r.i.y $w.f.v.r.i.l3 $w.f.v.r.i.z $w.f.v.r.i.l4 -side left
	pack $w.f.v.r.l $w.f.v.r.r $w.f.v.r.d $w.f.v.r.i -side top
	frame $w.f.v.t -relief groove -borderwidth 2
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
	pack $w.f.v.t.l $w.f.v.t.x $w.f.v.t.y $w.f.v.t.z -side top
	pack $w.f.v.r $w.f.v.t -side left
	frame $w.f.e
	label $w.f.e.l -text "RMS Error (cm): "
	label $w.f.e.e -textvariable $this-reg_error
	pack $w.f.e.l $w.f.e.e -side left -expand 1
	frame $w.f.b
	label $w.f.b.l1 -text "Auto Register:   "
	label $w.f.b.l2 -text "Iter#"
	global $this-curr_iter
	label $w.f.b.l3 -width 3 -textvariable $this-curr_iter
	button $w.f.b.auto -text "Go!" -command "$this-c auto"
	global $this-abortButton
	checkbutton $w.f.b.abort -text "Abort" -variable $this-abortButton
	global $this-newtonB
	checkbutton $w.f.b.newton -text "Newton" -variable $this-newtonB
	pack $w.f.b.l1 $w.f.b.l2 $w.f.b.l3 $w.f.b.auto $w.f.b.abort $w.f.b.newton -side left -expand 1
	frame $w.f.p
	label $w.f.p.l -text "% of Nodes (int) to use in auto-reg: "
	entry $w.f.p.e -relief sunken -width 3 -textvariable $this-percent
	pack $w.f.p.l $w.f.p.e -side left -expand 1
	make_labeled_radio $w.f.m "Error Metrix" "" \
		left $this-error_metric \
		{{"Blocks" blocks} \
		{"Dist^2" dist2} \
		{"Dist" dist}}
	frame $w.f.t 
	label $w.f.t.l -text "Error Tolerance: "
	entry $w.f.t.e -relief sunken -width 8 -textvariable $this-tolerance
	pack $w.f.t.l $w.f.t.e -side left -expand 1
	frame $w.f.i
	label $w.f.i.l -text "Max Iterations: "
	entry $w.f.i.e -relief sunken -width 4 -textvariable $this-iters
	pack $w.f.i.l $w.f.i.e -side left -expand 1
	frame $w.f.mgd
	button $w.f.mgd.bld -text "Build Full MGD" -command "$this-c build_full_mgd"
	button $w.f.mgd.print -text "Print MGD" -command "$this-c print_mgd"
	pack $w.f.mgd.bld $w.f.mgd.print -side left -expand 1
	pack $w.f.v $w.f.e $w.f.b $w.f.p $w.f.m $w.f.t $w.f.i $w.f.mgd -side top
	pack $w.f
    }
}
