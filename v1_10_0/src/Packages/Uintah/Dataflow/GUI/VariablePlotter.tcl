#
#  VariablePlotter.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   May 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl_class Uintah_Visualization_VariablePlotter {
    inherit Module

    protected var_list ""
    protected mat_lists {}
    protected type_list ""
    protected var_val_list {}
    protected time_list {}
    protected num_colors 240

    protected matrix_types {"Determinant" "Trace" "Norm"}
    protected vector_types {"length" "length2" "x" "y" "z"}
    protected num_m_type
    protected num_v_type
    # this represents the number of graphs and tables made
    # when a new graph or table is created then this number is incremented
    # Thus repeatidly punching graph or table will continue to make new ones
    # without replacing old ones.
    protected display_data_id 0

    constructor {config} {
	set name VariablePlotter
	set_defaults
    }
    
    method set_defaults {} {
	puts "In VariablePlotter"
	global $this-nl
	set $this-nl 0

	#plane selection
	global $this-var_orientation
	set $this-var_orientation 0
	global $this-index_l
	set $this-index_l 0
	global $this-index_x
	set $this-index_x 0
	global $this-index_y
	set $this-index_y 0
	global $this-index_z
	set $this-index_z 0

	#var graphing stuff
	global $this-curr_var
	set var_list ""

	# selection stuff
	set num_m_type [llength $matrix_types]
	set num_v_type [llength $vector_types]
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
	$this makeFrames $w
    }
    method build {} {
	set w .ui[modname]

	$this makeFrames $w
    }
    method destroyFrames {} {
	set w .ui[modname]
	destroy $w.vars
	set $var_list ""
    }
    method makeFrames {w} {
	$this buildVarFrame $w
    }
    method graph {name} {
    }
    # this sets all the variables, var_root_j, with val
    method mat_sel_sub { var_root number val} {
	for {set j 0} { $j < $number} {incr j} {
	    set tail "_$j"
	    set $var_root$tail $val
	}
    }
    # called when SelAll or SelNone is evoked from the top level
    method mat_sel { var_root mat_list val type} {
	for {set i 0} { $i < [llength $mat_list] } {incr i} {
	    set j [lindex $mat_list $i]
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
    method make_mat_menu {w mat_list var_id type} {
	set fname "$w.mat$var_id"
	menubutton $fname -text "Material" \
		-menu $fname.list -relief groove
	pack $fname -side right -padx 2 -pady 2
	
	menu $fname.list
	set var_o $var_id
	append var_o "_o[set $this-var_orientation]"
	$fname.list add command -label "Sel All" \
		-command "$this mat_sel $this-matvar_$var_o {$mat_list} 1 $type"
	$fname.list add command -label "Sel None" \
		-command "$this mat_sel $this-matvar_$var_o {$mat_list} 0 $type"
	for {set i 0} { $i < [llength $mat_list]} {incr i} {
	    set j [lindex $mat_list $i]
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
    # this function extracts which variables to get information from the global
    # variables set up with the widget.
    #
    # displaymethod - should be either graph or table, this value is passed on
    #                 to the c code which will call the appropiate function to
    #                 display the information
    # name - the name of the variable to graph, this is used by the c code to
    #        extract the values from the data archive
    # var_index - the index into the internal data structures that correspond
    #             to the variable "name"
    # mat_list - the list of material indecies, this is used to recreate the
    #            global variables used to determine if the variable should be
    #            graphed
    # type - the type of the variable, should be either matrix3, vector, or
    #        scaler
    method extract {displaymethod name var_index mat_list type} {
	set val_list {}
	set num_vals 0
	set var_root $this-matvar_$var_index
	append var_root "_o[set $this-var_orientation]"
#	puts "var_root = $var_root"
	# loop over all the materials	
	for {set i 0} { $i < [llength $mat_list]} {incr i} {
	    set j [lindex $mat_list $i]
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
	    set call "$this-c extract_data $displaymethod $name $var_index $num_vals"
	    for {set i 0} { $i < [llength $val_list] } { incr i } {
		set insert [lindex $val_list $i]
		append call " $insert"
	    }
#	    puts "call =  $call"
	    eval $call
	}
    }
    method addVar {w name mat_list type i} {
	set fname "$w.var$i"
	frame $fname
	pack $fname -side top -fill x -padx 2 -pady 2

	label $fname.label -text "$name"
	pack $fname.label -side left -padx 2 -pady 2

	button $fname.table -text "Table" \
		-command "$this extract table $name $i {$mat_list} $type"
	pack $fname.table -side right -padx 2 -pady 2

	button $fname.graph -text "Graph" \
		-command "$this extract graph $name $i {$mat_list} $type"
	pack $fname.graph -side right -padx 2 -pady 2

	make_mat_menu $fname $mat_list $i $type
    }
    method setVar_list { args } {
	set var_list $args
#	puts "var_list is now $var_list"
    }
    method clearMat_list {} {
	set mat_lists {}
#	puts "mat_lists cleared"
    }
    method appendMat_list { args } {
	set mat_list $args
	lappend mat_lists $mat_list
#	puts "mat_list is now $mat_lists"
    }
    method setType_list { args } {
	set type_list $args
#	puts "type_list is now $type_list"
    }
    method buildVarFrame {w} {
	if {[llength $var_list] > 0} {
	    frame $w.vars -borderwidth 3 -relief ridge
	    pack $w.vars -side top -fill x -padx 2 -pady 2
	    
#	    puts "var_list length [llength $var_list]"
	    for {set i 0} { $i < [llength $var_list] } { incr i } {
		set newvar [lindex $var_list $i]
		set newmat_list [lindex $mat_lists $i]
		set newtype [lindex $type_list $i]
		addVar $w.vars $newvar $newmat_list $newtype $i
	    }
	}
    }
    method do_nothing {} {
#	puts "l = [set $this-index_l]"
#	puts "x = [set $this-index_x]"
#	puts "y = [set $this-index_y]"
#	puts "z = [set $this-index_z]"
#	$this-c "pick"
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
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 190
	set n "$this-c needexecute "
	set pick "$this-c update_sn "

	#selection stuff
	frame $w.o
	pack $w.o -side top -fill x -padx 2 -pady 2

	frame $w.o.select
	pack $w.o.select -side left -fill x -padx 2 -pady 2
	
	radiobutton $w.o.select.node -variable $this-var_orientation \
		-command $n -text "Node Centered" -value 0
	pack $w.o.select.node -side top -anchor w -pady 2 -ipadx 3

	radiobutton $w.o.select.cell -variable $this-var_orientation \
		-command $n -text "Cell Centered" -value 1
	pack $w.o.select.cell -side top -anchor w -pady 2 -ipadx 3
	
	# node ID
	make_entry $w.o.nodel "level index:" $this-index_l $pick
	pack $w.o.nodel -side top -fill x -padx 2 -pady 2
	make_entry $w.o.nodex "x index:" $this-index_x $pick
	pack $w.o.nodex -side top -fill x -padx 2 -pady 2
	make_entry $w.o.nodey "y index:" $this-index_y $pick
	pack $w.o.nodey -side top -fill x -padx 2 -pady 2
	make_entry $w.o.nodez "z index:" $this-index_z $pick
	pack $w.o.nodez -side top -fill x -padx 2 -pady 2

	makeFrames $w

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
#	button $w.ttest -text "Table test" -command "$this table_test"
#	pack $w.ttest -side bottom -expand yes -fill x
#	button $w.gtest -text "Graph test" -command "$this graph_test"
#	pack $w.gtest -side bottom -expand yes -fill x
    }
    method reset_var_val {} {
	set var_val_list {}
    }
    method set_time { args } {
	set time_list $args
	puts "time_list =  $time_list"
    }
    method set_var_val { args } {
	set val_list $args
	lappend var_val_list $val_list
	puts "Args to set_var_val were $args"
#	puts "New var_val_list: $var_val_list"
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
    method graph_data { id var pointname args } {
	set w .graph[modname]$display_data_id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
	incr display_data_id
#	wm minsize $w 300 300

#	puts "id = $id"
#	puts "var = $var"
#	puts "args = $args"

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Plot of $var at $pointname" \
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
	
#==========TESTING==========*/
#	if { ($max - $min) > 1000 || ($max - $min) < 1e-3 } {
#	    $w.graph yaxis configure -logscale true -title $var
#	} else {
#	    $w.graph yaxis configure -title $var
#	}

      $w.graph yaxis configure -title $var 
#==========TESTING==========`*/	

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
    method table_data { id var pointname args } {
	set w .table[modname]$display_data_id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
	incr display_data_id

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -fill x
	
	#seperate the materials from the types
	set args_mat {}
	set args_type {}
	for {set i 0} { $i < [llength $args] } {incr i} {
	    lappend args_mat [lindex $args $i]
	    incr i
	    lappend args_type [lindex $args $i]
	}

	# create the scrolled frame
	iwidgets::scrolledframe $w.sf -width 300 -height 300 \
		-labeltext "Data for $var at $pointname" \
		-vscrollmode dynamic -hscrollmode dynamic \
		-sbwidth 10

	# get the childsite to add table stuff to
	set cs [$w.sf childsite]
	blt::table $cs

	# set up columns for time idicies
	blt::table $cs [label $cs.time_list_title -text "TimeStep\nIndex"] \
		0,0
	blt::table $cs [label $cs.time_list_title2 -text "TimeStep\nNumber"] \
		0,1
	set time_list_length [llength $time_list]
	for { set i 0 } { $i < $time_list_length } {incr i} {
	    # the array index
	    blt::table $cs [label $cs.time_index$i -text $i] [expr $i+1],0 
	    # the actual time step
	    blt::table $cs [label $cs.time_value$i -text [lindex $time_list $i]] [expr $i+1],1
	}
	
	# now add all the variables
	set vvlist_length [llength $var_val_list]
	for { set i 0 } { $i < $vvlist_length } {incr i} {
	    # extract the values for this variable
	    set mat_index  [lindex $args_mat $i]
	    set mat_type [lindex $args_type $i]
	    set mat_vals [lindex $var_val_list $i]

	    
	    set line_name "Material_$mat_index"
	    if {$mat_type != "invalid"} {
		append line_name "_$mat_type"
	    }
	    
	    set column [expr $i + 2]
	    blt::table $cs [label $cs.top$line_name -text $line_name] 0,$column
	    # a for loop for each time step
	    set mat_vals_length [llength $mat_vals]
	    for { set t 0 } { $t < $mat_vals_length } {incr t} {
		set box_name "val$line_name"
		append box_name "_$t"
		blt::table $cs [label $cs.$box_name -text [lindex $mat_vals $t]] [expr $t+1],$column
	    }
	}

	pack $w.sf -fill both -expand yes -padx 10 -pady 10
    }

    # This is test code used to create a window with a scrollable blt table
    # This code is not normally accessable from the main user interface
    method table_test {} {
	set w .table[modname]test
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w

	iwidgets::scrolledframe $w.sf -width 60 -height 50 \
		-labeltext scrolledframe \
		-vscrollmode dynamic -hscrollmode dynamic \
		-sbwidth 10
#		-labelmargin 0 -scrollmargin 0

	set cs [$w.sf childsite]
	blt::table $cs \
		[label $cs.title -text "A Table"] 0,0 -cspan 3\
		[button $cs.10 -text OK] 1,0 -fill both \
		[label $cs.11 -text 3.1415] 1,1 -fill y \
		[label $cs.12 -text 4.5e3] 1,2 -fill x \
		[button $cs.20 -text NOTOK] 2,0 -fill both \
		[label $cs.21 -text 4.5123123] 2,1 -fill y \
		[label $cs.22 -text 0.54234134134] 2,2 -fill x

#	frame $w.table_frame

	pack $w.sf -expand yes -fill both -padx 10 -pady 10

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -fill x
	
    }

    # This is test code used to create a window with a graph
    # This code is not normally accessable from the main user interface
    method graph_test {} {
	set w .graph[modname]test
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
#	wm minsize $w 300 300

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Test graph" -height 250 \
		-plotbackground gray99

	set x1 {0 1 2 3 4 5}
	set y1 {0 1 2 3 4}
	set x2 {0 1 2 3 4 5}
	set y2 {4 3 2 1 0}
	set x3 {0 1 2 3 4 5}
	set y3 {1 4 2 0 3}

	$w.graph yaxis configure -title "y-values"
	$w.graph xaxis configure -title "x-values" -loose true

	$w.graph element create line1 -linewidth 2 -pixels 3 \
		-color [$this get_color 0]  -xdata $x1 -ydata $y1
	$w.graph element create line2 -linewidth 2 -pixels 3 \
		-color [$this get_color 20] -xdata $x2 -ydata $y2
	$w.graph element create line3 -linewidth 2 -pixels 3 \
		-color [$this get_color 40] -xdata $x3 -ydata $y3

	$w.graph element create line4 -linewidth 2 -pixels 3 \
		-color [$this get_color 60]  -xdata $x1 -ydata $y1
	$w.graph element create line5 -linewidth 2 -pixels 3 \
		-color [$this get_color 80] -xdata $x2 -ydata $y2
	$w.graph element create line6 -linewidth 2 -pixels 3 \
		-color [$this get_color 100] -xdata $x3 -ydata $y3

	$w.graph element create line7 -linewidth 2 -pixels 3 \
		-color [$this get_color 120]  -xdata $x1 -ydata $y1
	$w.graph element create line8 -linewidth 2 -pixels 3 \
		-color [$this get_color 140] -xdata $x2 -ydata $y2
	$w.graph element create line9 -linewidth 2 -pixels 3 \
		-color [$this get_color 160] -xdata $x3 -ydata $y3

	$w.graph element create line10 -linewidth 2 -pixels 3 \
		-color [$this get_color 180]  -xdata $x1 -ydata $y1
	$w.graph element create line11 -linewidth 2 -pixels 3 \
		-color [$this get_color 200] -xdata $x2 -ydata $y2
	$w.graph element create line12 -linewidth 2 -pixels 3 \
		-color [$this get_color 220] -xdata $x3 -ydata $y3

	$w.graph element create line13 -linewidth 2 -pixels 3 \
		-color [$this get_color 240]  -xdata $x1 -ydata $y1
	$w.graph element create line14 -linewidth 2 -pixels 3 \
		-color [$this get_color 280] -xdata $x2 -ydata $y2
	$w.graph element create line15 -linewidth 2 -pixels 3 \
		-color [$this get_color 1100] -xdata $x3 -ydata $y3

	pack $w.graph
    }
}	
	
    

    









