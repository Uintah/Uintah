########################################
#CLASS
#    VizControl
#    Visualization control for simulation data that contains
#    information on both a regular grid in particle sets.
#OVERVIEW TEXT
#    This module receives a ParticleGridReader object.  The user
#    interface is dynamically created based information provided by the
#    ParticleGridReader.  The user can then select which variables he/she
#    wishes to view in a visualization.
#KEYWORDS
#    ParticleGridReader, Material/Particle Method
#AUTHOR
#    Kurt Zimmerman
#    Department of Computer Science
#    University of Utah
#    January 1999
#    Copyright (C) 1999 SCI Group
#LOG
#    Created January 5, 1999
########################################

catch {rename ParticleFieldExtractor ""}

itcl_class Uintah_Visualization_ParticleFieldExtractor { 
    inherit Module 

    protected psVarList ""
    protected pvVarList ""
    protected ptVarList ""
    
    protected pf ""

################################################
    protected var_list ""
# mat_list is not used because all variables are assumed to have 
# num_materials materials.
#    protected mat_list ""
    protected type_list ""
    protected var_val_list {}
    protected graph_data_names ""
    protected time_list {}
    protected num_materials 0
    protected num_colors 240

    protected matrix_types {"Determinant" "Trace" "Norm"}
    protected vector_types {"length" "length2" "x" "y" "z"}
    protected num_m_type
    protected num_v_type
    # this represents the number of graphs made
    # when a new graph is created then this number is incremented
    # Thus repeatidly punching graph will continue to make new ones
    # without replacing old ones.
    protected graph_id 0

###############################################

    constructor {config} { 
        set name ParticleFieldExtractor 
        set_defaults

    } 

    method set_defaults {} { 
        global $this-tcl_status 
	global $this-psVar;
	global $this-pvVar;
	global $this-ptVar;
	global $this-pNMaterials;
	set $this-psVar ""
	set $this-pvVar ""
	set $this-ptVar ""
	set $this-pName ""

	# selection stuff
	set num_m_type [llength $matrix_types]
	set num_v_type [llength $vector_types]
    } 
    
    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
        } else { 
	    $this buildTopLevel
	    wm deiconify $w
            raise $w 
	}
    }

    method buildTopLevel {} {
        set w .ui[modname] 

        if {[winfo exists $w]} { 
            return;
        } 
	
        toplevel $w 
	wm withdraw $w
	
	set n "$this-c needexecute"
	frame $w.f -relief flat
 	pack $w.f -side top -expand yes -fill both

	button $w.b -text "Close" -command "wm withdraw $w"
	pack $w.b -side top -fill x -padx 2 -pady 2

	makeFrames $w.f
    }


    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }
    
    method Rebuild {} {
	set w .ui[modname]

	$this destroyFrames
	$this makeFrames $w.f
    }

    method build {} {
	set w .ui[modname]

	$this makeFrames $w.f
    }

    method makeFrames { parent } {
	frame $parent.f0 -relief groove -borderwidth 2 
	label $parent.f0.label -text Particles
	pack $parent.f0 -side left -expand yes -fill both -padx 2
	pack $parent.f0.label -side top
	frame $parent.f0.1 -relief flat -borderwidth 2
	pack $parent.f0.1 -side top -expand yes -fill both -padx 2
	buildControlFrame $parent.f0.1

	set pf $parent.f0
    }
    

    method buildControlFrame { name } {

	if { [winfo exists $name.1] } {
	    destroy $name.1
	}
	if { [winfo exists $name.2] } {
	    destroy $name.2
	}
	frame $name.1 -relief flat -borderwidth 2
	label $name.1.s -text "Scalars"
	pack $name.1 -side left  -expand yes -fill both -padx 2
	pack $name.1.s -side top
	frame $name.2 -relief flat -borderwidth 2
	label $name.2.v -text "Vectors"
	pack $name.2 -side left -expand yes -fill both -padx 2
	pack $name.2.v -side top
	frame $name.3 -relief flat -borderwidth 2
	label $name.3.v -text "Tensors"
	pack $name.3 -side left -expand yes -fill both -padx 2
	pack $name.3.v -side top
    }

    method destroyFrames {} {
	set w .ui[modname] 

	destroy $pf
	set pNameList ""
	set psVarList ""
	set pvVarList ""
	set ptVarList ""
	set pf ""
    }

    method setParticleScalars { args } {
	set psVarList $args;
	puts "psVarList is now $psVarList";
    }
    method setParticleVectors { args } {
	set pvVarList $args;
	puts "pvVarList is now $pvVarList";
    }    
    method setParticleTensors { args } {
	set ptVarList $args;
	puts "ptVarList is now $ptVarList";
    }    
 
    method buildPMaterials { ns } {
	set parent $pf
	set buttontype checkbutton
	set c "$this-c needexecute"

	frame $parent.m -relief flat -borderwidth 2
	pack $parent.m -side top
	label $parent.m.l -text Material
	pack $parent.m.l -side top
	frame $parent.m.m -relief flat 
	pack $parent.m.m  -side top -expand yes -fill both
	for {set i 0} { $i < $ns} {incr i} {
	    $buttontype $parent.m.m.p$i -text $i \
		-offvalue 0 -onvalue 1 -command $c \
		-variable $this-p$i
	    pack $parent.m.m.p$i -side left
	    if { $i == 0 } {
		$parent.m.m.p$i select
	    }
	    #puts [$parent.m.p$i configure -variable]
	}
	set num_materials $ns
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }

    method buildVarList {} {
	global $this-pName
	set sv ""
	set vv ""
	set c "$this-c needexecute"
	puts "... buildControlFrame $pf.1"
	#set varlist [split $pvVarList]
	for {set i 0} { $i < [llength $psVarList] } { incr i } {
	    set newvar [lindex $psVarList $i]
	    if { $i == 0 && [set $this-psVar] == ""} {
		set $this-psVar $newvar
	    }
	    set lvar [string tolower $newvar]
	    regsub \\. $lvar _ lvar
	    puts "button $pf.1.1.$lvar"
	    radiobutton $pf.1.1.$lvar -text $newvar \
		-variable $this-psVar -command $c -value $newvar
	    pack $pf.1.1.$lvar -side top -anchor w
	    if { $newvar == [set $this-psVar] } {
		$pf.1.1.$lvar invoke
	    }
	}
	## set the Particle Vector Variables
	#set varlist [split $pvVarList]
	for {set i 0} { $i < [llength $pvVarList] } { incr i } {
	    set newvar [lindex $pvVarList $i]
	    if { $i == 0 && [set $this-pvVar] == ""} {
		set $this-pvVar $newvar
	    }
	    set lvar [string tolower $newvar]
	    regsub \\. $lvar _ lvar
	    radiobutton $pf.1.2.$lvar -text $newvar \
		-variable $this-pvVar -command $c -value $newvar
	    pack $pf.1.2.$lvar -side top -anchor w
	    if { $newvar == [set $this-pvVar] } {
		$pf.1.2.$lvar invoke
	    }
	}
	for {set i 0} { $i < [llength $ptVarList] } { incr i } {
	    set newvar [lindex $ptVarList $i]
	    if { $i == 0 && [set $this-ptVar] == ""} {
		set $this-ptVar $newvar
	    }
	    set lvar [string tolower $newvar]
	    regsub \\. $lvar _ lvar
	    radiobutton $pf.1.3.$lvar -text $newvar \
		-variable $this-ptVar -command $c -value $newvar
	    pack $pf.1.3.$lvar -side top -anchor w
	    if { $newvar == [set $this-ptVar] } {
		$pf.1.3.$lvar invoke
	    }
	}
    }      
	
    method infoFrame { id } {
	set w .info[modname]$id
        if {[winfo exists $w]} { 
            destroy $w 
        } 

	toplevel $w
	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x
	
    }

    method infoAdd { id addbutton args } {
	set w .info[modname]$id
        if { ![winfo exists $w]} { 
            $this infoFrame $id 
        }
	

	set text [join $args]
	set var [lindex $args 0]
	set args1 [string tolower $var ]
	frame $w.$args1 -relief flat 
	pack $w.$args1 -side top -expand yes -fill x
	label $w.$args1.l -text $text
	pack $w.$args1.l -anchor w -side left   -expand yes -fill x
	if { $addbutton } {
	    button $w.$args1.b -text "Graph $var over time" \
		-command "$this-c graph $id $var"
	    pack $w.$args1.b -side left -expand yes -fill x
	}
    }

#################################################################3
    method create_part_graph_window { part_id } {
	# var_list,mat_list(or num_materials), and time_list should
	# be initialized by this point
	# part_id must not have any periods in it

        set w .ui[modname]$part_id

	#$this setVar_list "p.mass" "p.stress" "p.cheese"
	#$this setMat_list 3 3 3
	#$this setTime_list 0 1 2 3 4
	
        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
        } else { 
	    # build the window
	    toplevel $w
	    $this buildVarFrame $w

	    wm deiconify $w
            raise $w 
	}
    }
    method setVar_list { args } {
	set var_list $args
	puts "var_list is now $var_list"
    }
    method setMat_list { args } {
	puts "Warning setMat_list has not effect!!"
#	set mat_list $args
#	puts "mat_list is now $mat_list"
    }
    method setType_list { args } {
	set type_list $args
	puts "type_list is now $type_list"
    }
    method setTime_list { args } {
	set time_list $args
	puts "time_list is now $time_list"
    }
    method reset_var_val {} {
	set var_val_list {}
    }
    method set_var_val { args } {
	set val_list $args
	lappend var_val_list $val_list
	puts "Args were $args"
	puts "New var_val_list: $var_val_list"
    }
    method setGraph_names { args } {
	set graph_data_names $args
	puts "graph_data_names is now $graph_data_names"
    }
    method mat_sel_b { var num_mat val} {
	for {set j 0} { $j < $num_mat} {incr j} {
	    set nvar $var
	    append nvar $j
	    set $this-matvar$nvar $val
	}
    }
    method make_mat_menu_b {w mat i} {
	set fname "$w.mat$mat"
	menubutton $fname -text "Material" \
		-menu $fname.list -relief groove
	pack $fname -side right -padx 2 -pady 2
	
	menu $fname.list
	$fname.list add command -label "Sel All" \
		-command "$this mat_sel $i $mat 1"
	$fname.list add command -label "Sel None" \
		-command "$this mat_sel $i $mat 0"
	for {set j 0} { $j < $mat} {incr j} {
	    set var $i
	    append var $j
	    set $this-matvar$var 0
	    $fname.list add checkbutton \
		    -variable $this-matvar$var \
		    -label "Mat $j"
	}
    }
    method graphbutton_b {name var_index num_mat} {
#	$this-c graph $name [set $this-vmat$i] $i
	set val_list {}
	set num_vals 0
	for {set j 0} { $j < $num_mat} {incr j} {
	    set nvar $var_index
	    append nvar $j
	    if {[set $this-matvar$nvar] != 0} {
		lappend val_list $j
		incr num_vals
	    }
	}
	puts "Calling $this-c graph"
	puts "name      = $name"
	puts "num_mat   = $num_mat"
	puts "var_index = $var_index"
	puts "num_vals  = $num_vals"
	puts "val_list  = $val_list"
	set call "$this-c graph $name $var_index $num_vals"
	for {set i 0} { $i < [llength $val_list] } { incr i } {
	    set insert [lindex $val_list $i]
	    append call " $insert"
	}
	eval $call
	puts "call = $call"
    }
    method addVar_b {w name mat i} {
	set fname "$w.var$i"
	frame $fname
	pack $fname -side top -fill x -padx 2 -pady 2

	label $fname.label -text "$name"
	pack $fname.label -side left -padx 2 -pady 2

	button $fname.button -text "Graph" -command "$this graphbutton $name $i $mat"
	pack $fname.button -side right -padx 2 -pady 2

	if {$mat > $num_materials} {
	    set num_materials $mat
	    puts "num_materials is now $num_materials"
	}

	make_mat_menu $fname $mat $i
    }
    method buildVarFrame_b {w } {
	if {[llength $var_list] > 0} {
	    frame $w.vars -borderwidth 3 -relief ridge
	    pack $w.vars -side top -fill x -padx 2 -pady 2
	    
	    puts "var_list length [llength $var_list]"
	    for {set i 0} { $i < [llength $var_list] } { incr i } {
		set newvar [lindex $var_list $i]
#		set newmat [lindex $mat_list $i]
#		addVar $w.vars $newvar $newmat $i
		addVar $w.vars $newvar $num_materials $i
	    }
	}
    }

    method graph_data_b { id var args } {
	set w .graph[modname]$id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
#	wm minsize $w 300 300

	puts "id = $id"
	puts "var = $var"
	puts "args = $args"

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Plot of $var with materials $args" \
		-height 250 -plotbackground gray99

	set max 1e-10
	set min 1e+10

	puts "length of var_val_list = [llength $var_val_list]"
	puts "length of args         = $args"
	for { set i 0 } { $i < [llength $args] } {incr i} {
	    set mat_vals [lindex $var_val_list [lindex $args $i]]
	    for { set j 0 } { $j < [llength $mat_vals] } {incr j} {
		set val [lindex $mat_vals $j]
		if { $max < $val } { set max $val }
		if { $min > $val } { set min $val }
	    }
	}
	
	if { ($max - $min) > 1000 || ($max - $min) < 1e-3 } {
	    $w.graph yaxis configure -logscale true -title $var
	} else {
	    $w.graph yaxis configure -title $var
	}
	
	$w.graph xaxis configure -title "Timestep" -loose true
	
	puts "length of var_val_list = [llength $var_val_list]"
	for { set i 0 } { $i < [llength $args] } {incr i} {
	    puts "adding line"
	    set mat_index  [lindex $args $i]
	    set mat_vals [lindex $var_val_list $mat_index]
	    puts "\[llength mat_vals\] = [llength $mat_vals]"
	    puts "mat_vals = $mat_vals"
	    set color_ind [expr round(double($mat_index) / ($num_materials-1) * ($num_colors - 1))]
	    $w.graph element create "Material $mat_index" -linewidth 2 \
		-pixels 3 -color [$this get_color $color_ind] \
		-xdata $time_list -ydata $mat_vals
	}
	
	pack $w.graph
    }

    method get_color { index } {
	set color_scheme {
	    { 255 0 0}  { 255 102 0}
	    { 255 204 0}  { 255 234 0}
	    { 204 255 0}  { 102 255 0}
	    { 0 255 0}    { 0 255 102}
	    { 0 255 204}  { 0 204 255}
	    { 0 102 255}  { 0 0 255}}
	#set color_scheme { {255 0 0} {0 255 0} {0 0 255} }
	set incr {}
	set upper_bounds [expr [llength $color_scheme] -1]
	for {set j 0} { $j < $upper_bounds} {incr j} {
	    set c1 [lindex $color_scheme $j]
	    set c2 [lindex $color_scheme [expr $j + 1]]
	    set incr_a {}
	    lappend incr_a [expr [lindex $c2 0] - [lindex $c1 0]]
	    lappend incr_a [expr [lindex $c2 1] - [lindex $c1 1]]
	    lappend incr_a [expr [lindex $c2 2] - [lindex $c1 2]]
	    lappend incr $incr_a
	}
	lappend incr {0 0 0}
#	puts "incr = $incr"
	set step [expr $num_colors / [llength $color_scheme]]
	set ind [expr $index % $num_colors] 
	set i [expr $ind / $step]
	set im [expr double($ind % $step)/$step]
#	puts "i = $i  im = $im"
	set curr_color [lindex $color_scheme $i]
	set curr_incr [lindex $incr $i]
#	puts "curr_color = $curr_color, curr_incr = $curr_incr"
	set r [expr [lindex $curr_color 0]+round([lindex $curr_incr 0] * $im)] 
	set g [expr [lindex $curr_color 1]+round([lindex $curr_incr 1] * $im)] 
	set b [expr [lindex $curr_color 2]+round([lindex $curr_incr 2] * $im)] 
	set c [format "#%02x%02x%02x" $r $g $b]
#	puts "r=$r, g=$g, b=$b, c=$c"
	return $c
    }
###############################################################################
    method buildVarFrame {w} {
	if {[llength $var_list] > 0} {
	    frame $w.vars -borderwidth 3 -relief ridge
	    pack $w.vars -side top -fill x -padx 2 -pady 2
	    
#	    puts "var_list length [llength $var_list]"
	    for {set i 0} { $i < [llength $var_list] } { incr i } {
		set newvar [lindex $var_list $i]
#		set newmat [lindex $mat_list $i]
		set newtype [lindex $type_list $i]
		addVar $w.vars $newvar $num_materials $newtype $i
	    }
	}
    }
    # this sets all the variables, var_root_j, with val
    method mat_sel_sub { var_root number val} {
	for {set j 0} { $j < $number} {incr j} {
	    set tail "_$j"
	    set $var_root$tail $val
	}
    }
    # called when SelAll or SelNone is evoked from the top level
    method mat_sel { var_root number val type} {
	for {set j 0} { $j < $number} {incr j} {
	    set tail "_$j"
	    switch $type {
		"matrix3" {
		    mat_sel_sub $var_root$tail $num_m_type $val
		}
		"vector" {
		    mat_sel_sub $var_root$tail $num_v_type $val
		}
		"scaler" {
		    set $var_root$tail $val
		}
	    }
	}
    }	
    # creates the material selection menu
    # it generates sub menus for matrix3 and vector types
    method make_mat_menu {w mat i type} {
	set fname "$w.mat$i"
	menubutton $fname -text "Material" \
		-menu $fname.list -relief groove
	pack $fname -side right -padx 2 -pady 2
	
	menu $fname.list
	set var_o $i
#	append var_o "_o[set $this-var_orientation]"
	$fname.list add command -label "Sel All" \
		-command "$this mat_sel $this-matvar_$var_o $mat 1 $type"
	$fname.list add command -label "Sel None" \
		-command "$this mat_sel $this-matvar_$var_o $mat 0 $type"
	for {set j 0} { $j < $mat} {incr j} {
	    set var $var_o
	    append var "_$j"
#	    puts "***var = $var"
	    if {$type == "matrix3"} {
		$fname.list add cascade -label "Mat $j" \
			-menu $fname.list.types$var
		menu $fname.list.types$var
		$fname.list.types$var add command -label "Sel All" \
			-command "$this mat_sel_sub $this-matvar_$var \
			$num_m_type 1"
		$fname.list.types$var add command -label "Sel None" \
			-command "$this mat_sel_sub $this-matvar_$var \
			$num_m_type 0"
		for {set k 0} { $k < $num_m_type} {incr k} {
		    set var2 $var
		    append var2 "_$k"
		    $fname.list.types$var add checkbutton \
			    -variable $this-matvar_$var2 \
			    -label [lindex $matrix_types $k]
		}
	    } elseif {$type == "vector"} {
		$fname.list add cascade -label "Mat $j" \
			-menu $fname.list.types$var
		menu $fname.list.types$var
		$fname.list.types$var add command -label "Sel All" \
			-command "$this mat_sel_sub $this-matvar_$var \
			$num_v_type 1"
		$fname.list.types$var add command -label "Sel None" \
			-command "$this mat_sel_sub $this-matvar_$var \
			$num_v_type 0"
		for {set k 0} { $k < $num_v_type} {incr k} {
		    set var2 $var
		    append var2 "_$k"
		    $fname.list.types$var add checkbutton \
			    -variable $this-matvar_$var2 \
			    -label [lindex $vector_types $k]
		}
	    } elseif {$type == "scaler"} {
		$fname.list add checkbutton \
			-variable $this-matvar_$var \
			-label "Mat $j"
	    }
	}
    }
    method graphbutton {name var_index num_mat type} {
	set val_list {}
	set num_vals 0
	set var_root $this-matvar_$var_index
#	append var_root "_o[set $this-var_orientation]"
#	puts "var_root = $var_root"
	# loop over all the materials	
	for {set j 0} { $j < $num_mat} {incr j} {
	    set mat_root $var_root
	    append mat_root "_$j"
	    switch $type {
		"matrix3" {
		    for {set k 0} { $k < $num_m_type} {incr k} {
			set tail "_$k"	
			if {[set $mat_root$tail] != 0} {
			    lappend val_list "$j" [lindex $matrix_types $k]
			    incr num_vals
			}
		    }
		}
		"vector" {
		    for {set k 0} { $k < $num_v_type} {incr k} {
			set tail "_$k"
			if {[set $mat_root$tail] != 0} {
			    lappend val_list "$j" [lindex $vector_types $k]
			    incr num_vals
			}
		    }
		}
		"scaler" {
		    if {[set $mat_root] != 0} {
			lappend val_list "$j" "invalid"
			incr num_vals
		    }
		}
	    }
	}
#	puts "Calling $this-c graph"
#	puts "name      = $name"
#	puts "num_mat   = $num_mat"
#	puts "var_index = $var_index"
#	puts "num_vals  = $num_vals"
#	puts "val_list  = $val_list"
	if {[llength $val_list] > 0} {
	    set call "$this-c graph $name $var_index $num_vals"
	    for {set i 0} { $i < [llength $val_list] } { incr i } {
		set insert [lindex $val_list $i]
		append call " $insert"
	    }
#	    puts "call =  $call"
	    eval $call
	}
    }
    method addVar {w name mat type i} {
	set fname "$w.var$i"
	frame $fname
	pack $fname -side top -fill x -padx 2 -pady 2

	label $fname.label -text "$name"
	pack $fname.label -side left -padx 2 -pady 2

	button $fname.button -text "Graph" -command "$this graphbutton $name $i $mat $type"
	pack $fname.button -side right -padx 2 -pady 2

	if {$mat > $num_materials} {
	    set num_materials $mat
#	    puts "num_materials is now $num_materials"
	}

	make_mat_menu $fname $mat $i $type
    }
    method graph_data { id var args } {
	set w .graph[modname]$graph_id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
	incr graph_id
#	wm minsize $w 300 300

#	puts "id = $id"
#	puts "var = $var"
#	puts "args = $args"

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Plot of $var" \
		-height 250 -plotbackground gray99

	set max 1e-10
	set min 1e+10

	#seperate the materials from the types
	set args_mat {}
	set args_type {}
	for {set i 0} { $i < [llength $args] } {incr i} {
	    lappend args_mat [lindex $args $i]
	    incr i
	    lappend args_type [lindex $args $i]
	}
#	puts "args_mat = $args_mat"
#	puts "args_type = $args_type"
	
#	puts "length of var_val_list = [llength $var_val_list]"
#	puts "length of args_mat     = $args_mat"
	for { set i 0 } { $i < [llength $var_val_list] } {incr i} {
	    set mat_vals [lindex $var_val_list $i]
#	    puts "mat_vals = $mat_vals"
	    for { set j 0 } { $j < [llength $mat_vals] } {incr j} {
		set val [lindex $mat_vals $j]
		if { $max < $val } { set max $val }
		if { $min > $val } { set min $val }
	    }
	}
	
	if { ($max - $min) > 1000 || ($max - $min) < 1e-3 } {
	    $w.graph yaxis configure -logscale true -title $var
	} else {
	    $w.graph yaxis configure -title $var
	}
	
	$w.graph xaxis configure -title "Timestep" -loose true
	
	set vvlist_length [llength $var_val_list]
#	puts "length of var_val_list = [llength $var_val_list]"
	for { set i 0 } { $i < $vvlist_length } {incr i} {
#	    puts "adding line"
	    set mat_index  [lindex $args_mat $i]
	    set mat_type [lindex $args_type $i]
	    set mat_vals [lindex $var_val_list $i]
	    if {$i == 0} {
		set color_ind 0
	    } else {
		set color_ind [expr round(double($i) / \
			($vvlist_length-1) * ($num_colors -1))]
	    }
	    #	    set mat_vals [lindex $var_val_list $mat_index]
	    #	    set color_ind [expr round(double($mat_index) / ($num_materials-1) * ($num_colors - 1))]
	    set line_name "Material_$mat_index"
	    if {$mat_type != "invalid"} {
		append line_name "_$mat_type"
	    }
	    $w.graph element create $line_name -linewidth 2 -pixels 3 \
		    -color [$this get_color $color_ind] \
		    -xdata $time_list -ydata $mat_vals
	}
	
	pack $w.graph
    }

	    	    
}


