##
 #  AnisoSphereModel.tcl:
 #
 #  Author: Sascha Moehrs
 #
 ##

package require Iwidgets 3.0

catch {rename AnisoSphereModel ""}

itcl_class BioPSE_Forward_AnisoSphereModel {
    inherit Module
    constructor {config} {
        set name AnisoSphereModel
        set_defaults
    }
    method set_defaults {} {
		# radii
		global $this-r_scalp
		set $this-r_scalp 0.075
		global $this-r_skull
		set $this-r_skull 0.071
		global $this-r_cbsf
		set $this-r_cbsf 0.065
		global $this-r_brain
		set $this-r_brain 0.063
		
		# radii unit
		global $this-units
		set $this-units "dm"

		# radial conductivities
		global $this-rc_scalp
		set $this-rc_scalp 0.33
		global $this-rc_skull
		set $this-rc_skull 0.0042
		global $this-rc_cbsf
		set $this-rc_cbsf 1.0
		global $this-rc_brain
		set $this-rc_brain 0.33

		# tangential conductivities
		global $this-tc_scalp
		set $this-tc_scalp 0.33
		global $this-tc_skull
		set $this-tc_skull 0.0042
		global $this-tc_cbsf
		set $this-tc_cbsf 1.0
		global $this-tc_brain
		set $this-tc_brain 0.33
    }
    
    method ui {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w
		wm minsize $w 300 300

		frame $w.f 
		pack $w.f -padx 2 -pady 2 -expand 1 -fill x

		# radii
		iwidgets::labeledframe $w.f.lf -labelpos nw -labeltext "sphere radii"
		set lfr [$w.f.lf childsite]

		global $this-r_scalp
		label $lfr.laScRa -text "scalp: "
		entry $lfr.enScRa -width 20 -textvariable $this-r_scalp 
		bind  $lfr.enScRa <Return> "$this-c needexecute"
		grid  $lfr.laScRa  -row 0 -column 0 -sticky e
		grid  $lfr.enScRa  -row 0 -column 1 -columnspan 2 -sticky "ew"

		global $this-r_skull
		label $lfr.laSkRa -text "skull: "
		entry $lfr.enSkRa -width 20 -textvariable $this-r_skull
		bind  $lfr.enSkRa <Return> "$this-c needexecute"
		grid  $lfr.laSkRa -row 1 -column 0 -sticky e
		grid  $lfr.enSkRa -row 1 -column 1 -columnspan 2 -sticky "ew"


		global $this-r_cbsf
		label $lfr.laCbsfRa -text "cbsf: "
		entry $lfr.enCbsfRa -width 20 -textvariable $this-r_cbsf
		bind  $lfr.enCbsfRa <Return> "$this-c needexecute"
		grid  $lfr.laCbsfRa -row 2 -column 0 -sticky e
		grid  $lfr.enCbsfRa -row 2 -column 1 -columnspan 2 -sticky "ew"

		global $this-r_brain
		label $lfr.laBrRa -text "brain: "
		entry $lfr.enBrRa -width 20 -textvariable $this-r_brain
		bind  $lfr.enBrRa <Return> "$this-c needexecute"
		grid  $lfr.laBrRa -row 3 -column 0 -sticky e
		grid  $lfr.enBrRa -row 3 -column 1 -columnspan 2 -sticky "ew"

		global $this-units
		label $lfr.laUn -text "unit: "
		entry $lfr.enUn -width 20 -textvariable $this-units
		bind  $lfr.enUn <Return> "$this-c needexecute"
		grid  $lfr.laUn -row 4 -column 0 -sticky "e"
		grid  $lfr.enUn -row 4 -column 1 -columnspan 2 -sticky "ew"

		grid columnconfigure . 1 -weight 1

		pack $w.f.lf -side top -fill x -expand 1

		# radial conductivities
		iwidgets::labeledframe $w.f.rc -labelpos "nw" -labeltext "radial conductivities \[ S / m \]"
		set rcc [$w.f.rc childsite]

		global $this-rc_scalp
		label $rcc.laScRC -text "scalp: "
		entry $rcc.enScRC -width 20 -textvariable $this-rc_scalp
		bind  $rcc.enScRC <Return> "$this-c needexecute"
		grid  $rcc.laScRC -row 0 -column 0 -sticky e
		grid  $rcc.enScRC -row 0 -column 1 -columnspan 2 -sticky "ew"

		global $this-rc_skull
		label $rcc.laSkRC -text "skull: "
		entry $rcc.enSkRC -width 20 -textvariable $this-rc_skull
		bind  $rcc.enSkRC <Return> "$this-c needexecute"
		grid  $rcc.laSkRC -row 1 -column 0 -sticky e
		grid  $rcc.enSkRC -row 1 -column 1 -columnspan 2 -sticky "ew"
		
		global $this-rc_cbsf
		label  $rcc.laCbsfRC -text "cbsf: "
		entry  $rcc.enCbsfRC -width 20 -textvariable $this-rc_cbsf
		bind   $rcc.enCbsfRC <Return> "$this-c needexecute"
		grid   $rcc.laCbsfRC -row 2 -column 0 -sticky e
		grid   $rcc.enCbsfRC -row 2 -column 1 -columnspan 2 -sticky "ew"

		global $this-rc_brain
		label  $rcc.laBrRC -text "brain: "
		entry  $rcc.enBrRC -width 20 -textvariable $this-rc_brain
		bind   $rcc.enBrRC <Return> "$this-c needexecute"
		grid   $rcc.laBrRC -row 3 -column 0 -sticky e
		grid   $rcc.enBrRC -row 3 -column 1 -columnspan 2 -sticky "ew"

		grid columnconfigure . 1 -weight 1

		pack $w.f.rc -side top -fill x -expand 1

		# tangential conductivities
		iwidgets::labeledframe $w.f.tc -labelpos "nw" -labeltext "tangential conductivities \[ S / m \]"
		set tcc [$w.f.tc childsite]

		global $this-tc_scalp
		label $tcc.laScTC -text "scalp: "
		entry $tcc.enScTC -width 20 -textvariable $this-tc_scalp
		bind  $tcc.enScTC <Return> "$this-c needexecute"
		grid  $tcc.laScTC -row 0 -column 0 -sticky e
		grid  $tcc.enScTC -row 0 -column 1 -columnspan 2 -sticky "ew"

		global $this-tc_skull
		label  $tcc.laSkTC -text "skull: "
		entry  $tcc.enSkTC -width 20 -textvariable $this-tc_skull
		bind   $tcc.enSkTC <Return> "$this-c needexecute"
		grid   $tcc.laSkTC -row 1 -column 0 -sticky e
		grid   $tcc.enSkTC -row 1 -column 1 -columnspan 2 -sticky "ew"

		global $this-tc_cbsf
		label  $tcc.laCbsfTC -text "cbsf: "
		entry  $tcc.enCbsfTC -width 20 -textvariable $this-tc_cbsf
		bind   $tcc.enCbsfTC <Return> "$this-c needexecute"
		grid   $tcc.laCbsfTC -row 2 -column 0 -sticky e
		grid   $tcc.enCbsfTC -row 2 -column 1 -columnspan 2 -sticky "ew"

		global $this-tc_brain
		label  $tcc.laBrTC -text "brain: "
		entry  $tcc.enBrTC -width 20 -textvariable $this-tc_brain
		bind   $tcc.enBrTC <Return> "$this-c needexecute"
		grid   $tcc.laBrTC -row 3 -column 0 -sticky e
		grid   $tcc.enBrTC -row 3 -column 1 -columnspan 2 -sticky "ew"

		grid columnconfigure . 1 -weight 1

		pack $w.f.tc -side top -fill x -expand 1

    }
}
