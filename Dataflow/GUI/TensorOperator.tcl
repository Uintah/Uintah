
itcl_class Uintah_Operators_TensorOperator {
    inherit Module

    method set_defaults {} {
	global $this-operation
	set $this-operation 0
	
	# element extractor
	global $this-row
	global $this-column
	global $this-elem
	set $this-row 2
	set $this-column 2
	set $this-elem 5

	# 2D eigen evaluator
	global $this-planeSelect
	global $this-delta
	global $this-eigen2D-calc-type
	set $this-planeSelect 2
	set $this-eigen2D-calc-type 0
	set $this-delta 1

    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left -anchor w
	entry $w.e -textvariable $v -width 8
	bind $w.e <Return> $c
	pack $w.e -side right -anchor e
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
 	toplevel $w

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -expand yes -fill x -padx 2 -pady 2

	frame $w.calc -relief raised -bd 1
	label $w.calc.l -text "Select Operation"
	radiobutton $w.calc.elem -text "Extract Element" \
		-variable $this-operation -value 0 \
		-command "$this select_element_extractor"
	#radiobutton $w.calc.eigen -text "Eigen-value/vector" \
	#	-variable $this-operation -value 1 \
	#	-command "$this select_eigen_evaluator"
	radiobutton $w.calc.eigen2D -text "2D Eigenvalues" \
		-variable $this-operation -value 1 \
		-command "$this select_eigen2D"
	radiobutton $w.calc.pressure -text "Pressure" \
		-variable $this-operation -value 2 \
		-command "$this select_pressure"
	radiobutton $w.calc.eqivstress -text "Equivalent Stress" \
		-variable $this-operation -value 3 \
		-command "$this select_equivalent_stress"
	radiobutton $w.calc.octshearstress -text "Octahedral Shear Stress" \
		-variable $this-operation -value 4 \
		-command "$this select_oct_shear_stress"
	pack $w.calc.l $w.calc.elem $w.calc.eigen2D $w.calc.pressure $w.calc.eqivstress $w.calc.octshearstress -anchor w
	pack $w.calc -side left -padx 2 -pady 2 -fill y

	if { [set $this-operation] == 0} {
	    select_element_extractor
	} elseif { [set $this-operation] == 1} {
	    select_eigen2D
	} elseif { [set $this-operation] == 2} {
	    select_pressure
	} elseif { [set $this-operation] == 3} {
	    select_equivalent_stress
	} elseif { [set $this-operation] == 4} {
	    select_oct_shear_stress
	}
    }

    method select_element_extractor {} {
	set w .ui[modname]
	destroy $w.opts
	frame $w.opts -relief sunken -bd 1
	element_extractor_ui $w.opts
	pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	$this-c needexecute
    }
    #method select_eigen_evaluator {} {
	#set w .ui[modname]
	#destroy $w.opts
	#frame $w.opts -relief sunken -bd 1
	#eigen_ui $w.opts
	#pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	#$this-c needexecute
    #}
    method select_eigen2D {} {
	set w .ui[modname]
	destroy $w.opts
	frame $w.opts -relief sunken -bd 1
	eigen2D_ui $w.opts
	pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	$this-c needexecute
    }
    method select_pressure {} {
	set w .ui[modname]
	destroy $w.opts
	frame $w.opts -relief sunken -bd 1
	pressure_ui $w.opts
	pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	$this-c needexecute
    }
    method select_equivalent_stress {} {
	set w .ui[modname]
	destroy $w.opts
	frame $w.opts -relief sunken -bd 1
	equivalent_stress_ui $w.opts
	pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	$this-c needexecute
    }
    method select_oct_shear_stress {} {
	set w .ui[modname]
	destroy $w.opts
	frame $w.opts -relief sunken -bd 1
	oct_shear_stress_ui $w.opts
	pack $w.opts -padx 2 -pady 2 -fill y -expand yes
	$this-c needexecute
    }

    method element_extractor_ui {w} {
	set n "$this element_extractor_select"

	frame $w.m
	pack $w.m -padx 2 -pady 2 -fill x -expand yes

	frame $w.m.r1
	radiobutton $w.m.r1.c1 -command $n -variable $this-elem -value 1
	radiobutton $w.m.r1.c2 -command $n -variable $this-elem -value 2
	radiobutton $w.m.r1.c3 -command $n -variable $this-elem -value 3
	pack $w.m.r1.c1 $w.m.r1.c2 $w.m.r1.c3 -side left -anchor n
	frame $w.m.r2
	radiobutton $w.m.r2.c1 -command $n -variable $this-elem -value 4
	radiobutton $w.m.r2.c2 -command $n -variable $this-elem -value 5
	radiobutton $w.m.r2.c3 -command $n -variable $this-elem -value 6
	pack $w.m.r2.c1 $w.m.r2.c2 $w.m.r2.c3 -side left -anchor n
	frame $w.m.r3
	radiobutton $w.m.r3.c1 -command $n -variable $this-elem -value 7
	radiobutton $w.m.r3.c2 -command $n -variable $this-elem -value 8
	radiobutton $w.m.r3.c3 -command $n -variable $this-elem -value 9
	pack $w.m.r3.c1 $w.m.r3.c2 $w.m.r3.c3 -side left -anchor n

	pack $w.m.r1 $w.m.r2 $w.m.r3

	make_entry $w.row "Row" $this-row "$this element_extractor_enter"
	make_entry $w.column "Column" $this-column "$this element_extractor_enter"
	pack $w.row $w.column -expand yes -fill x -padx 5 -pady 2
    }
    method element_extractor_select {} {
	set $this-row [expr ([set $this-elem] + 2) / 3]
	set $this-column [expr ([set $this-elem] - 1) % 3 + 1]

	$this-c needexecute
    }   
    method element_extractor_enter {} {
	set $this-row [expr ([set $this-row] - 1) % 3 + 1]
	set $this-column [expr ([set $this-column] - 1) % 3 + 1]
	set $this-elem [expr 3*([set $this-row]-1) + [set $this-column]]

	$this-c needexecute
    }

    #method eigen_ui {w} {
	#set n "$this-c needexecute"
	#make_labeled_radio $w.r "Which Eigen Value" $n top \
	#	$this-eigenSelect {{"Largest" 0} {"Middle" 1} {"Smallest" 2}}
	#pack $w.r -anchor c -expand yes
    #}

    method pressure_ui {w} {
	label $w.l1 -text "Stress Tensor Operation"
	label $w.l2 -text "P = (-stress11-stress22-stress33)/3"
	pack $w.l1
	pack $w.l2 -anchor c -expand yes
    }

    method equivalent_stress_ui {w} {
	label $w.l1 -text "Stress Tensor Operation"
	label $w.l2 -text "s_eq = sqrt(1.5*(sdev_ij*sdev_ij))"
	pack $w.l1 
	pack $w.l2 -anchor c -expand yes
    }

    method oct_shear_stress_ui {w} {
	label $w.l1 -text "Octahedral Shear Stress Tensor Operation"
	label $w.l2 -text \
	"OSS = sqrt( (stress00-stress11)*(stress00-stress11)+
(stress11-stress22)*(stress11-stress22)+
(stress22-stress00)*(stress22-stress00)+
6*(stress01*stress01+stress12*stress12+stress02*stress02))
____________________________________________
            3.0"

        pack $w.l1 
	pack $w.l2 -anchor c -expand yes
    }

    method eigen2D_ui {w} {
	set n "$this-c needexecute"

	make_labeled_radio $w.rp "Plane" $n left \
		$this-planeSelect {{"xy" 2} {"xz" 1} {"yz" 0}}

	frame $w.calc
	label $w.calc.l -text "Calculation"
	radiobutton $w.calc.r1 -text "|e2 - e1|" \
		-variable $this-eigen2D-calc-type -value 0 \
		-command "$this eigen2D_absDiffCalc"
	radiobutton $w.calc.r2 -text "cos(|e2 - e1| / Delta)" \
		-variable $this-eigen2D-calc-type -value 1 \
		-command "$this eigen2D_cosDiffCalc"
	pack $w.calc.l -anchor w
	pack $w.calc.r1 $w.calc.r2 -side left

	make_entry $w.delta "Delta" $this-delta $n

	pack $w.rp $w.calc $w.delta -anchor w -pady 5

	eigen2D_absDiffCalc
    }
    method eigen2D_absDiffCalc {} {
	set w .ui[modname]
	set color "#505050"
	$w.opts.delta.e configure -state disabled
	$w.opts.delta.e configure -foreground $color
	$w.opts.delta.l configure -foreground $color
	$this-c needexecute
    }
    method eigen2D_cosDiffCalc {} {
	set w .ui[modname]
	$w.opts.delta.e configure -state normal
	$w.opts.delta.e configure -foreground black
	$w.opts.delta.l configure -foreground black
	$this-c needexecute
    }

}
