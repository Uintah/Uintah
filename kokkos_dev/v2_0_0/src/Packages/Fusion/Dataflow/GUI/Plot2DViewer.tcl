package require Iwidgets 3.0  

itcl_class Fusion_Render_Plot2DViewer {
    inherit Module

    constructor {config} {
        set name Plot2DViewer
        set_defaults
    }

    method set_defaults {} {

	global $this-havePLplot

	global haveData

	global $this-updateType

	global $this-nPlots
	global nTabs
	global active_tab

	global $this-nData

	global $this-xmin
	global $this-xmax
	global $this-ymin
	global $this-ymax
	global $this-zmin
	global $this-zmax

	set $this-havePLplot 0

	set haveData 0

	set $this-updateType 0

	set $this-nPlots 1
	set nTabs 0
	set active_tab 0

	set $this-nData 0
    }

    method have_data { flag } {
	global haveData

	set haveData $flag
    }

    method ui {} {

	global $this-havePLplot

	$this-c have_PLplot

	if { [set $this-havePLplot] == 0 } {
	    set w .ui[modname]
	    if {[winfo exists $w]} {
		wm deiconify $w
		raise $w
	    } else {
		toplevel $w
		
		label $w.label \
		    -text "PLplot is not availible with this build."  \
		    -width 40 -anchor w -just l

		pack $w.label
	    }	    
	    
	    return
	}

        set w .ui[modname]
        if {[winfo exists $w]} {
	    wm deiconify $w
            raise $w
	    return
        } else {
	    toplevel $w

	    # Plotting window.
	    PLWin $w.plw

	    # Main ui which will be detachable.
	    frame $w.ui

	    # Plot tabs window with in the main ui window.
	    iwidgets::labeledframe $w.ui.tab_title \
		-labelpos nw -labeltext "Plots"

	    set plot [$w.ui.tab_title childsite]

	    iwidgets::tabnotebook  $plot.tabs -height 400  -width 500 -raiseselect true 

	    for {set i 0} {$i < [set $this-nPlots]} {incr i} {
		add_tab $plot.tabs $i
	    }

	    $plot.tabs view "Plot 0"
	    $plot.tabs configure -tabpos "n"
	    pack $plot.tabs -side top -expand yes

	    # UI Widget window.
	    frame $w.ui.widgets

	    #  Options
	    global $this-updateType

	    iwidgets::labeledframe $w.ui.widgets.opt -labelpos nw -labeltext "Options"
	    set opt [$w.ui.widgets.opt childsite]
	    
	    iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this update_type $opt.update"
	    
	    $opt.update insert end Manual Auto
	    $opt.update select [set $this-updateType]

	    pack $opt.update

	    frame $w.ui.widgets.win

	    iwidgets::spinner $w.ui.widgets.win.n -labeltext "Plot Windows:" \
		-width 10 -fixed 10 \
		-validate "$this spin_in $plot.tabs $this-nPlots %P 1 8" \
		-decrement "$this spin_incr $plot.tabs $w.ui.widgets.win.n $this-nPlots -1 1 8" \
		-increment "$this spin_incr $plot.tabs $w.ui.widgets.win.n $this-nPlots +1 1 8" 

	    $w.ui.widgets.win.n insert 0 [set $this-nPlots]

	    pack $w.ui.widgets.win.n -side top


	    frame $w.ui.widgets.misc

	    button $w.ui.widgets.misc.update -text "Execute" \
		-command "$this needexecute 1 1"
	    button $w.ui.widgets.misc.close -text "Dismiss" \
		-command "destroy $w"

	    pack $w.ui.widgets.misc.update $w.ui.widgets.misc.close \
		-side left -padx 25

	    pack $w.ui.widgets.opt $w.ui.widgets.win $w.ui.widgets.misc\
		-side top -padx 10 -pady 5

	    pack $w.ui.widgets $w.ui.tab_title -side left -expand yes

	    pack $w.plw $w.ui -side top -padx 10 -pady 5
	}

	needexecute 1 0
    }

    method set_color { w plt dat } {
	global $w
	global $this-color-$plt-$dat

	switch -- [$w get] {
	    "Red"    { set $this-color-$plt-$dat 1 }
	    "Yellow" { set $this-color-$plt-$dat 2 }
	    "Green"  { set $this-color-$plt-$dat 3 }
	    "Cyan"   { set $this-color-$plt-$dat 4 }
	    "Gray"   { set $this-color-$plt-$dat 7 }
	    "Blue"   { set $this-color-$plt-$dat 9 }
	    "Purple" { set $this-color-$plt-$dat 10 }
	    "Orange" { set $this-color-$plt-$dat 14 }
	    "White"  { set $this-color-$plt-$dat 15 }
	    default  { set $this-color-$plt-$dat 0 }
	}
    }

    method update_type { w } {
	global $w
	global $this-updateType

	set $this-updateType [$w get]
    }

    method needexecute { updateGraph force } {
    
	if { $updateGraph == 1 } {
	    $this-c update_graph
	}

	if { [set $this-updateType] == "Auto" || $force == 1 } {
	    $this-c needexecute
	}
    }

    method add_tab { w plt } {

	global nTabs
	
	global tab-$plt

	set pl "Plot "
	append pl $plt

	set tab-$plt [$w add -label "$pl" \
			  -command "$this set_active_tab $plt"]
	add_tab_gui [set tab-$plt] $plt

	incr nTabs 1
    }

    method delete_tab { w plt } {
	global nTabs
	
	set pl "Plot "
	append pl $plt

	$w delete "$pl"

	incr nTabs -1

	$this-c remove_GUIVar "title-$plt"
	$this-c remove_GUIVar "abscissa-$plt"
	$this-c remove_GUIVar "ordinate-$plt"

	$this-c remove_GUIVar "dims-$plt"
	$this-c remove_GUIVar "style-$plt"
	$this-c remove_GUIVar "slice-$plt"
	$this-c remove_GUIVar "altitude-$plt"
	$this-c remove_GUIVar "azimuth-$plt"
    }

    method update_tabs { w } {

	global nTabs

	if { $nTabs < [set $this-nPlots] } {

	    for {set i $nTabs} {$i < [set $this-nPlots]} {incr i 1} {
		add_tab $w $i
	    }
	} elseif { $nTabs > [set $this-nPlots] } {

	    for {set i [expr $nTabs - 1]} {$i >= [set $this-nPlots]} {incr i -1} {
		delete_tab $w $i
	    }
	}

	needexecute 1 0
    }


    method set_active_tab {act} {
	global active_tab

	set active_tab $act
    }

    
    method add_tab_gui {w plt} {

	global $this-title-$plt
	global $this-abscissa-$plt
	global $this-ordinate-$plt

	global $this-dims-$plt
	global $this-style-$plt

	$this-c add_GUIVar_String "title-$plt" "Plot $plt"
	$this-c add_GUIVar_String "abscissa-$plt" X 
	$this-c add_GUIVar_String "ordinate-$plt" Y

	$this-c add_GUIVar_Int "style-$plt" 0 
	$this-c add_GUIVar_Int "altitude-$plt" 60
	$this-c add_GUIVar_Int "azimuth-$plt" 120

 	set $this-title-$plt "Plot $plt"
 	set $this-abscissa-$plt X
 	set $this-ordinate-$plt Y

 	set $this-dims-$plt 2
 	set $this-style-$plt 0
 	set $this-altitude-$plt 60
 	set $this-azimuth-$plt 120

	frame $w.plot

	frame $w.plot.style

	frame $w.plot.style.r

	make_labeled_radio $w.plot.style.r.d "Plot dimension:" "" top $this-dims-$plt \
	    {{"1D" 1} {"2D" 2} {"3D" 3}}

	make_labeled_radio $w.plot.style.r.s "Plot style:" "" top $this-style-$plt \
	    {{"Points" 0} {"Edges" 1} {"Edges - Culled" 2} {"Faces" 3} }

	pack $w.plot.style.r.d $w.plot.style.r.s -side left -padx 10 -pady 5 -fill y


	frame $w.plot.style.s

	iwidgets::spinner $w.plot.style.s.alt -labeltext "Altitude:" \
	    -width 10 -fixed 10 \
	    -validate  "$this spin_in   $w.plots.style.s $this-altitude-$plt %P 0 90" \
	    -decrement "$this spin_incr $w.plots.style.s $w.plot.style.s.alt $this-altitude-$plt -5 0 90" \
	    -increment "$this spin_incr $w.plots.style.s $w.plot.style.s.alt $this-altitude-$plt +5 0 90" 
	
	$w.plot.style.s.alt insert 0 [set $this-altitude-$plt]

	iwidgets::spinner $w.plot.style.s.az -labeltext "Azimuth:" \
	    -width 10 -fixed 10 \
	    -validate  "$this spin_in   $w.plot.style.s $this-azimuth-$plt %P 0 360" \
	    -decrement "$this spin_incr $w.plot.style.s $w.plot.style.s.az $this-azimuth-$plt -5 0 360" \
	    -increment "$this spin_incr $w.plot.style.s $w.plot.style.s.az $this-azimuth-$plt +5 0 360" 
	
	$w.plot.style.s.az insert 0 [set $this-azimuth-$plt]

	pack $w.plot.style.s.alt $w.plot.style.s.az -side top -padx 10 -pady 5


	pack $w.plot.style.r $w.plot.style.s -side left -padx 10 -pady 5


	frame $w.plot.title

	labelEntry $w.plot.title.main     "Title"    5 $this-title-$plt    25

	frame $w.plot.title.axis
	labelEntry $w.plot.title.axis.abscissa "Abscissa" 8 $this-abscissa-$plt 15
	labelEntry $w.plot.title.axis.ordinate "Ordinate" 8 $this-ordinate-$plt 15
	pack $w.plot.title.axis.abscissa $w.plot.title.axis.ordinate -side left -padx 10 -pady 5

	pack $w.plot.title.main $w.plot.title.axis -side top -pady 2 -fill x


	pack $w.plot.style $w.plot.title -side top -padx 10 -pady 5

# Create a frame for the field data.
  	frame $w.data -relief groove -borderwidth 2
  	pack $w.data -side left -padx 2 -pady 2 -fill y
  	label $w.data.title -text "Data:"
  	pack $w.data.title -side top
  	canvas $w.data.canvas -width 400 -height 100 \
  	        -scrollregion "0 0 1000 800" \
  		-xscrollcommand "$w.data.xscroll set" -borderwidth 0 -xscrollincrement 10 \
  		-yscrollcommand "$w.data.yscroll set" -borderwidth 0 -yscrollincrement 10
	
  	frame $w.data.canvas.frame -relief sunken -borderwidth 2
  	pack $w.data.canvas.frame
  	$w.data.canvas create window 0 1 -window $w.data.canvas.frame \
  	    -anchor nw
	
  	scrollbar $w.data.xscroll -orient horizontal -relief sunken \
  	    -command "$w.data.canvas xview"
  	scrollbar $w.data.yscroll -orient vertical -relief sunken \
  	    -command "$w.data.canvas yview"

  	pack $w.data.yscroll -fill y -side left -padx 2 -pady 2

  	pack $w.data.xscroll -fill x -side bottom -padx 2 -pady 2

  	pack $w.data.canvas -side right -padx 2 -pady 2 -fill y

	global data-$plt

	set data-$plt $w.data.canvas.frame

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {

	    global $this-kdim-$dat
	    add_datum $plt $dat 0 [expr [set $this-kdim-$dat] - 1]
	}

 	pack $w.data $w.plot -side bottom -padx 10 -pady 5
    }

    method data_size { ndata } {

	for {set plt 0} {$plt < [set $this-nPlots]} {incr plt} {

	    set w .ui[modname]
 	    if [ expr [winfo exists $w] ] {
		
		global data-$plt

		set w [set data-$plt]

		for {set dat $ndata} {$dat < [set $this-nData]} {incr dat} {
		
		    pack forget $w.data$dat

		    $this-c remove_GUIVar "slice-$plt-$dat"
		    destroy  $w.data$dat
		}
	    }
	}

	set $this-nData $ndata
    }

    method add_data { dat idim jdim kdim } {
	for {set i 0} {$i < [set $this-nPlots]} {incr i} {

	    $this-c add_GUIVar_Int "idim-$dat" $idim
	    $this-c add_GUIVar_Int "jdim-$dat" $jdim
	    $this-c add_GUIVar_Int "kdim-$dat" $kdim

	    global $this-idim-$dat
	    global $this-jdim-$dat
	    global $this-kdim-$dat

	    set $this-idim-$dat $idim
	    set $this-jdim-$dat $jdim
	    set $this-kdim-$dat $kdim

	    add_datum $i $dat 0 [expr $kdim - 1]
	}
    }

    method add_datum { plt dat start stop } {
	    
	# Reset all of the slider values to the index values.
	set w .ui[modname]
	if [ expr [winfo exists $w] ] {
		
	    global data-$plt

	    $this-c add_GUIVar_Int "active-$plt-$dat" 0
	    $this-c add_GUIVar_Int "slice-$plt-$dat" 0
	    $this-c add_GUIVar_Int "skip-$plt-$dat" 1

	    global $this-active-$plt-$dat
	    global $this-color-$plt-$dat

	    global $this-slice-$plt-$dat
	    global $this-slice2-$plt-$dat

	    global $this-skip-$plt-$dat
	    global $this-skip2-$plt-$dat
	    
	    set $this-active-$plt-$dat 0
	    set $this-color-$plt-$dat 0

	    set $this-slice-$plt-$dat 0
	    set $this-slice2-$plt-$dat "0"

	    set $this-skip-$plt-$dat 1
	    set $this-skip2-$plt-$dat "1"

	    set w [set data-$plt]

	    if [ expr [winfo exists $w.data$dat] ] {
		return
	    } else {
		frame $w.data$dat

		label $w.data$dat.title -text "Field $dat"

		checkbutton $w.data$dat.active -text "" \
		    -variable $this-active-$plt-$dat

		# Skip slider
		frame $w.data$dat.skip

		label $w.data$dat.skip.title \
		    -text "Skip:"  \
		    -width 5 -anchor w -just left

		scaleEntry2 $w.data$dat.skip.index \
		    1 25 50 \
		    $this-skip-$plt-$dat $this-skip2-$plt-$dat

		pack $w.data$dat.skip.title -side left
		pack $w.data$dat.skip.index -side left


		frame $w.data$dat.slice

		# Slice slider
		if { $start != $stop } {
		    label $w.data$dat.slice.title \
			-text "Slice:"  \
			-width 6 -anchor w -just left

		    scaleEntry2 $w.data$dat.slice.index \
			$start $stop  200 \
			$this-slice-$plt-$dat $this-slice2-$plt-$dat

		    pack $w.data$dat.slice.title -side left
		    pack $w.data$dat.slice.index -side left
		}

		# Color Menu
		iwidgets::optionmenu $w.data$dat.color -labeltext "Color:" \
		    -labelpos w \
		    -command "$this set_color $w.data$dat.color $plt $dat"
	    
		$w.data$dat.color insert end Variable Red Green Blue Orange Yellow Purple Cyan White Gray
		$w.data$dat.color select [set $this-color-$plt-$dat]

		pack $w.data$dat.title $w.data$dat.active $w.data$dat.color \
		    $w.data$dat.skip $w.data$dat.slice -side left
		
		pack $w.data$dat -side top -fill x
	    }
	}
    }

    method graph_data { } {

	set w .ui[modname]
	if {[winfo exists $w.plw] != 1} {
	    return
	}

	# Plot layout based on the number of plots requested.
	matrix col int 9 = { 1, 1, 2, 2, 2, 3, 3, 4, 4 }
	matrix row int 9 = { 1, 1, 1, 2, 2, 2, 2, 2, 2 }

	# Set the plot window layout.
	$w.plw cmd plssub [col [set $this-nPlots]] [row [set $this-nPlots]]

	# Total number of plots.
	set nplots [expr [col [set $this-nPlots]] * [row [set $this-nPlots]]]

	# Clear all of the windows.
	for {set i 0} {$i < $nplots} {incr i} {
	    $w.plw cmd pladv 0
	}

	# Draw each plot based on the plot properties.
	for {set plt 0} {$plt < [set $this-nPlots]} {incr plt} {

	    global $this-dims-$plt
	    global $this-style-$plt

	    if { [set $this-dims-$plt] == 1 } {
		plot1D $w.plw $plt [set $this-style-$plt]
	    } elseif { [set $this-dims-$plt] == 2 } {
		plot2D $w.plw $plt [set $this-style-$plt]
	    } else {
		plot3D $w.plw $plt [set $this-style-$plt]
	    }
	}    

	#	x01 $w.plw
	#	x08 $w.plw
	#	x16 $w.plw
    }

    method spin_in {w arg value min max} {
	if {! [regexp "\\A\\d*\\.*\\d+\\Z" $value]} {
	    return 0
	} elseif {$value < $min || $max < $value} {
	    return 0
	} 
	set $arg $value

	update_tabs $w

	return 1
    }

    method spin_incr {w sw arg incr min max} {
	set newarg [expr [set $arg] + $incr]

	if {$newarg > $max} {
	    set newarg [expr $newarg - $max]
	} elseif {$newarg < $min} {
	    set newarg [expr $max + $newarg]
	}   
	set $arg $newarg
	$sw delete 0 end
	$sw insert 0 [set $arg]

	update_tabs $w
    }

    method labelEntry { win text1 l1 text2 l2 } {
	frame $win 
	pack $win -side top -padx 5 -pady 5
	label $win.l -text $text1  -width $l1 -anchor w -just left
	label $win.c  -text ":" -width 1 -anchor w -just left 
	entry $win.e -text $text2 -width $l2 -just left -fore darkred
	pack $win.l $win.c -side left
	pack $win.e -padx 5 -side left
    }

    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
#	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]

	needexecute 1 0
    }

    method manualSliderEntry { start stop var1 var2 } {

	if { [set $var2] < $start } {
	    set $var2 $start
	}
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]

	needexecute 1 0
    }

    method set_size {dat idim jdim kdim} {

	global $this-idim-$dat
	global $this-jdim-$dat
	global $this-kdim-$dat
	
	set $this-idim-$dat $idim
	set $this-jdim-$dat $jdim
	set $this-kdim-$dat $kdim

	for {set plt 0} {$plt < [set $this-nPlots]} {incr plt} {

	    if { 0 < [expr $kdim - 1] } {

		global $this-slice-$plt-$dat
		global $this-slice2-$plt-$dat

		# Reset all of the slider values to the index values.
		set w .ui[modname]
		if [ expr [winfo exists $w] ] {
		    
		    global data-$plt
		    
		    set w [set data-$plt]

		    # Update the sliders to have the new end values.
		    $w.data$dat.slice.index.s \
			configure -from 0 -to [expr $kdim - 1]

		    bind $w.data$dat.slice.index.e \
			<Return> "$this manualSliderEntry 0 [expr $kdim - 1] $this-slice-$plt-$dat $this-slice2-$plt-$dat"
		}

		# Update the text values.
		set $this-slice2-$plt-$dat [set $this-slice-$plt-$dat]
	    }
	}
    }

    # {w loopback} = the plplot window
    # plt = the plot number
    # dat = the field port number
    # fill = the plotting style

    method plot1D {{w loopback} plt fill } {

	global $this-xmin
	global $this-xmax
	global $this-ymin
	global $this-ymax
	global $this-zmin
	global $this-zmax

	set xmin  10000000.0
	set xmax -10000000.0
	set ymin  10000000.0
	set ymax -10000000.0
	set zmin  10000000.0
	set zmax -10000000.0

	set plotSomething 0

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 } {

		global $this-idim-$dat
		global $this-jdim-$dat

		global haveData

		set haveData 0

# Matrices with the xy cordinates plus the value at that coordinate.
		matrix x-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix y-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix v-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]

# Get the values from the c++ code.
		$this-c vertex_coords $dat $this-slice-$plt-$dat \
		    x-$dat y-$dat v-$dat

# Make sure the data was retrived properly.
		if { $haveData != 1 } return

# Get the min max for this data set.
		if { $xmin > [set $this-xmin] } { set xmin [set $this-xmin] }
		if { $xmax < [set $this-xmax] } { set xmax [set $this-xmax] }
		if { $ymin > [set $this-ymin] } { set ymin [set $this-ymin] }
		if { $ymax < [set $this-ymax] } { set ymax [set $this-ymax] }
		if { $zmin > [set $this-zmin] } { set zmin [set $this-zmin] }
		if { $zmax < [set $this-zmax] } { set zmax [set $this-zmax] }

		set plotSomething 1
	    }
	}

	if { $plotSomething == 0 } return

#Create a rainbow color map.
	rainbow_cmap1 $w 36 1
	if { [set $this-zmax] > [set $this-zmin] } {
	    set cOffset [expr 1.0/([set $this-zmax]-[set $this-zmin])]
	} else {
	    set cOffset 0
	}

# Select color 1 from colormap 0
	$w cmd plcol0 1

# Set the plotting envelope i.e. the min-max of the plot.
	if { [set $this-xmax] == [set $this-xmin] } {
	    if { [set $this-xmax] > 0 } {
		set $this-xmax [expr [set $this-xmax] * 1.1]
	    } else {
		if { [set $this-xmax] < 0 } {
		    set $this-xmax [expr [set $this-xmax] * 0.9]
		} else {
		    if { [set $this-xmax] == 0 } {
			set $this-xmax 0.1
		    }
		}
	    }
	}

	if { [set $this-ymax] == [set $this-ymin] } {
	    if { [set $this-ymax] > 0 } {
		set $this-ymax [expr [set $this-ymax] * 1.1]
	    } else {
		if { [set $this-ymax] < 0 } {
		    set $this-ymax [expr [set $this-ymax] * 0.9]
		} else {
		    if { [set $this-ymax] == 0 } {
			set $this-ymax 0.1
		    }
		}
	    }
	}

	if { [set $this-zmax] == [set $this-zmin] } {
	    if { [set $this-zmax] > 0 } {
		set $this-zmax [expr [set $this-zmax] * 1.1]
	    } else {
		if { [set $this-zmax] < 0 } {
		    set $this-zmax [expr [set $this-zmax] * 0.9]
		} else {
		    if { [set $this-zmax] == 0 } {
			set $this-zmax 0.1
		    }
		}
	    }
	}

	$w cmd plenv [set $this-xmin] [set $this-xmax] \
  	             [set $this-ymin] [set $this-ymax] 0 0

	global $this-title-$plt
	global $this-abscissa-$plt
	global $this-ordinate-$plt

# Select color 2 from colormap 0
	$w cmd plcol0 5
	$w cmd pllab [set $this-abscissa-$plt] [set $this-ordinate-$plt] [set $this-title-$plt]

	matrix xPt float 1
	matrix yPt float 1

	matrix xPoly float 2
	matrix yPoly float 2

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 } {

		global $this-idim-$dat
		global $this-jdim-$dat

# Set the increment ammount
		global $this-skip-$plt-$dat
		set inc [set $this-skip-$plt-$dat]

# Draw a point for each node

		if { $fill == 0 } {

		    for {set i 0} {$i < [set $this-idim-$dat]} {incr i $inc} {
			
			xPt 0 = [x-$dat $i 0]
			yPt 0 = [v-$dat $i 0]

			if { [set $this-color-$plt-$dat] == 0 } {

			    set col [expr ([v-$dat $i 0]-[set $this-zmin])*$cOffset]
			    
			    if { 1.0 < $col } { set col 1.0 }
			    if { $col < 0.0 } { set col 0.0 }
			    
			    $w cmd plcol1 $col
			} else {
			    $w cmd plcol0 [set $this-color-$plt-$dat]
			}
# Plot a symbol 
			$w cmd plpoin 1 xPt yPt 1
		    }
		} else {
# Draw a line.
		    for {set i $inc} {$i < [set $this-idim-$dat]} {incr i $inc} {

			# Create a polygon for this index.
			set i1 [expr $i-$inc]
			
			xPoly 0 = [x-$dat $i1 0]
			yPoly 0 = [y-$dat $i1 0]

			xPoly 1 = [x-$dat $i 0]
			yPoly 1 = [y-$dat $i 0]

			if { [set $this-color-$plt-$dat] == 0 } {

			    set col [expr ([v-$dat $i 0]-[set $this-zmin])*$cOffset]
			    
			    if { 1.0 < $col } { set col 1.0 }
			    if { $col < 0.0 } { set col 0.0 }
			    
			    $w cmd plcol1 $col
			} else {

			    $w cmd plcol0 [set $this-color-$plt-$dat]
			}

			if { $fill == 1 || $fill == 2 } {
			    $w cmd plline 2 xPoly yPoly
			}
		    }
		}
	    }
	}
    }

    # {w loopback} = the plplot window
    # plt = the plot number
    # dat = the field port number
    # fill = the plotting style

    method plot2D {{w loopback} plt fill } {

	global $this-xmin
	global $this-xmax
	global $this-ymin
	global $this-ymax
	global $this-zmin
	global $this-zmax

	set xmin  10000000.0
	set xmax -10000000.0
	set ymin  10000000.0
	set ymax -10000000.0
	set zmin  10000000.0
	set zmax -10000000.0

	set plotSomething 0

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 } {

		global $this-idim-$dat
		global $this-jdim-$dat
		
		global haveData

		set haveData 0

# Matrices with the xy cordinates plus the value at that coordinate.
		matrix x-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix y-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix v-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]

# Get the values from the c++ code.
		$this-c vertex_coords $dat $this-slice-$plt-$dat \
		    x-$dat y-$dat v-$dat

# Make sure the data was retrived properly.
		if { $haveData != 1 } return

# Get the min max for this data set.
		if { $xmin > [set $this-xmin] } { set xmin [set $this-xmin] }
		if { $xmax < [set $this-xmax] } { set xmax [set $this-xmax] }
		if { $ymin > [set $this-ymin] } { set ymin [set $this-ymin] }
		if { $ymax < [set $this-ymax] } { set ymax [set $this-ymax] }
		if { $zmin > [set $this-zmin] } { set zmin [set $this-zmin] }
		if { $zmax < [set $this-zmax] } { set zmax [set $this-zmax] }

		set plotSomething 1
	    }
	}

	if { $plotSomething == 0 } return

#Create a rainbow color map.
	rainbow_cmap1 $w 36 1
	if { [set $this-zmax] > [set $this-zmin] } {
	    set cOffset [expr 1.0/([set $this-zmax]-[set $this-zmin])]
	} else {
	    set cOffset 0
	}

# Select color 1 from colormap 0
	$w cmd plcol0 1

# Set the plotting envelope i.e. the min-max of the plot.
	if { [set $this-xmax] == [set $this-xmin] } {
	    if { [set $this-xmax] > 0 } {
		set $this-xmax [expr [set $this-xmax] * 1.1]
	    } else {
		if { [set $this-xmax] < 0 } {
		    set $this-xmax [expr [set $this-xmax] * 0.9]
		} else {
		    if { [set $this-xmax] == 0 } {
			set $this-xmax 0.1
		    }
		}
	    }
	}

	if { [set $this-ymax] == [set $this-ymin] } {
	    if { [set $this-ymax] > 0 } {
		set $this-ymax [expr [set $this-ymax] * 1.1]
	    } else {
		if { [set $this-ymax] < 0 } {
		    set $this-ymax [expr [set $this-ymax] * 0.9]
		} else {
		    if { [set $this-ymax] == 0 } {
			set $this-ymax 0.1
		    }
		}
	    }
	}

	if { [set $this-zmax] == [set $this-zmin] } {
	    if { [set $this-zmax] > 0 } {
		set $this-zmax [expr [set $this-zmax] * 1.1]
	    } else {
		if { [set $this-zmax] < 0 } {
		    set $this-zmax [expr [set $this-zmax] * 0.9]
		} else {
		    if { [set $this-zmax] == 0 } {
			set $this-zmax 0.1
		    }
		}
	    }
	}

	$w cmd plenv [set $this-xmin] [set $this-xmax] \
  	             [set $this-ymin] [set $this-ymax] 0 0

	global $this-title-$plt
	global $this-abscissa-$plt
	global $this-ordinate-$plt

# Select color 2 from colormap 0
	$w cmd plcol0 5
	$w cmd pllab [set $this-abscissa-$plt] [set $this-ordinate-$plt] [set $this-title-$plt]


#	if { [set $this-idim-$dat] >= 1 &&
#	     [set $this-jdim-$dat] == 1 } {
#	    plot1D $w $plt $fill
#    }

	matrix xPt float 1
	matrix yPt float 1

	matrix xPoly float 5
	matrix yPoly float 5

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat
	    global $this-jdim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 &&
		 [set $this-jdim-$dat] >= 1 } {

# Set the increment ammount
		global $this-skip-$plt-$dat
		set inc [set $this-skip-$plt-$dat]

# Draw a point for each node

		if { $fill == 0 } {

		    for {set j 0} {$j < [set $this-jdim-$dat]} {incr j $inc} {
			for {set i 0} {$i < [set $this-idim-$dat]} {incr i $inc} {
			    
			    xPt 0 = [x-$dat $i $j]
			    yPt 0 = [y-$dat $i $j]

			    if { [set $this-color-$plt-$dat] == 0 } {

				set col [expr ([v-$dat $i $j]-[set $this-zmin])*$cOffset]	
				
				if { 1.0 < $col } { set col 1.0 }
				if { $col < 0.0 } { set col 0.0 }
				
				$w cmd plcol1 $col
			    } else {
				$w cmd plcol0 [set $this-color-$plt-$dat]
			    }
# Plot a symbol 
			    $w cmd plpoin 1 xPt yPt 1
			}
		    }
		} else {

# Draw a polygon or a filled polygon.

		    for {set j $inc} {$j < [set $this-jdim-$dat]} {incr j $inc } {
			for {set i $inc} {$i < [set $this-idim-$dat]} {incr i $inc} {

# Create a polygon for this index.
			    set i1 [expr $i-$inc]
			    set j1 [expr $j-$inc]
			    
			    xPoly 0 = [x-$dat $i1 $j1]
			    yPoly 0 = [y-$dat $i1 $j1]

			    xPoly 1 = [x-$dat $i $j1]
			    yPoly 1 = [y-$dat $i $j1]

			    xPoly 2 = [x-$dat $i $j]
			    yPoly 2 = [y-$dat $i $j]

			    xPoly 3 = [x-$dat $i1 $j]
			    yPoly 3 = [y-$dat $i1 $j]

			    xPoly 4 = [x-$dat $i1 $j1]
			    yPoly 4 = [y-$dat $i1 $j1]

			    if { [set $this-color-$plt-$dat] == 0 } {

				set col [expr ([v-$dat $i $j]-[set $this-zmin])*$cOffset]
				
				if { 1.0 < $col } { set col 1.0 }
				if { $col < 0.0 } { set col 0.0 }
				
				$w cmd plcol1 $col
			    } else {
				$w cmd plcol0 [set $this-color-$plt-$dat]
			    }

			    if { $fill == 1 || $fill == 2 } {
				$w cmd plline 5 xPoly yPoly
			    } elseif { $fill == 3 } {
				$w cmd plfill 4 xPoly yPoly
			    }
			}
		    }
		}
	    }
	}
    }

    # {w loopback} = the plplot window
    # plt = the plot number
    # dat = the field port number
    # fill = the plotting style

    method plot3D {{w loopback} plt fill} {

	global $this-xmin
	global $this-xmax
	global $this-ymin
	global $this-ymax
	global $this-zmin
	global $this-zmax

	set xmin  10000000.0
	set xmax -10000000.0
	set ymin  10000000.0
	set ymax -10000000.0
	set zmin  10000000.0
	set zmax -10000000.0

	set plotSomething 0

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 } {

		global $this-idim-$dat
		global $this-jdim-$dat

		global haveData

		set haveData 0

# Matrices with the xy cordinates plus the value at that coordinate.
		matrix x-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix y-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]
		matrix z-$dat float [set $this-idim-$dat] [set $this-jdim-$dat]

# Get the values from the c++ code.
		$this-c vertex_coords $dat $this-slice-$plt-$dat \
		    x-$dat y-$dat z-$dat

# Make sure the data was retrived properly.
		if { $haveData != 1 } return

# Get the min max for this data set.
		if { $xmin > [set $this-xmin] } { set xmin [set $this-xmin] }
		if { $xmax < [set $this-xmax] } { set xmax [set $this-xmax] }
		if { $ymin > [set $this-ymin] } { set ymin [set $this-ymin] }
		if { $ymax < [set $this-ymax] } { set ymax [set $this-ymax] }
		if { $zmin > [set $this-zmin] } { set zmin [set $this-zmin] }
		if { $zmax < [set $this-zmax] } { set zmax [set $this-zmax] }

		set plotSomething 1
	    }
	}

	if { $plotSomething == 0 } return

# Update the window index.
	$w cmd pladv 0

# Create a rainbow color map.
	rainbow_cmap1 $w 36 1
	if { [set $this-zmax] > [set $this-zmin] } {
	    set cOffset [expr 1.0/([set $this-zmax]-[set $this-zmin])]
	} else {
	    set cOffset 0
	}

# Select color 1 from colormap 0
	$w cmd plcol0 1

# Specify viewport using coordinates
	$w cmd plvpor 0.0 1.0 0.0 0.9
# Specify world coordinates of viewport boundaries
	$w cmd plwind -1.0 1.0 -0.9 1.1

	if { [set $this-xmax] == [set $this-xmin] } {
	    if { [set $this-xmax] > 0 } {
		set $this-xmax [expr [set $this-xmax] * 1.1]
	    } else {
		if { [set $this-xmax] < 0 } {
		    set $this-xmax [expr [set $this-xmax] * 0.9]
		} else {
		    if { [set $this-xmax] == 0 } {
			set $this-xmax 0.1
		    }
		}
	    }
	}

	if { [set $this-ymax] == [set $this-ymin] } {
	    if { [set $this-ymax] > 0 } {
		set $this-ymax [expr [set $this-ymax] * 1.1]
	    } else {
		if { [set $this-ymax] < 0 } {
		    set $this-ymax [expr [set $this-ymax] * 0.9]
		} else {
		    if { [set $this-ymax] == 0 } {
			set $this-ymax 0.1
		    }
		}
	    }
	}

	if { [set $this-zmax] == [set $this-zmin] } {
	    if { [set $this-zmax] > 0 } {
		set $this-zmax [expr [set $this-zmax] * 1.1]
	    } else {
		if { [set $this-zmax] < 0 } {
		    set $this-zmax [expr [set $this-zmax] * 0.9]
		} else {
		    if { [set $this-zmax] == 0 } {
			set $this-zmax 0.1
		    }
		}
	    }
	}

	$w cmd plw3d 1.0 1.0 1.0 \
	    [set $this-xmin] [set $this-xmax] \
	    [set $this-ymin] [set $this-ymax] \
	    [set $this-zmin] [set $this-zmax] \
	    [set $this-altitude-$plt] [set $this-azimuth-$plt]

# Draw a box with axes, etc, in 3-d
	$w cmd plcol0 2

	set title [format "Plot %d Alt=%.0f, Az=%.0f" \
		       $plt \
		       [set $this-altitude-$plt] \
		       [set $this-azimuth-$plt] ]
	$w cmd plmtex "t" 1.0 0.5 0.5 $title

	$w cmd plbox3 "bnstu" "R" 0.0 0 \
	    "bnstu" "Z" 0.0 0 \
	    "bcdmnstuv" "Pressure" 0.0 0

	matrix xPt float 1
	matrix yPt float 1
	matrix zPt float 1

	matrix xPoly float 5
	matrix yPoly float 5
	matrix zPoly float 5
	matrix zzPoly float 5 5
	matrix draw int 4 = { 1, 1, 1, 1 }

	for {set dat 0} {$dat < [set $this-nData]} {incr dat} {
	    
	    global $this-active-$plt-$dat
	    global $this-idim-$dat
	    global $this-jdim-$dat

	    # Make sure the data is active and there is data.
	    if { [set $this-active-$plt-$dat] == 1 &&
		 [set $this-idim-$dat] >= 1 &&
		 [set $this-jdim-$dat] >= 1 } {

# Set the increment ammount
		global $this-skip-$plt-$dat
		set inc [set $this-skip-$plt-$dat]

# Draw a point for each node

		if { $fill == 0 } {

		    for {set j 0} {$j < [set $this-jdim-$dat]} {incr j $inc} {
			for {set i 0} {$i < [set $this-idim-$dat]} {incr i $inc} {
			    
			    xPt 0 = [x-$dat $i $j]
			    yPt 0 = [y-$dat $i $j]
			    zPt 0 = [z-$dat $i $j]

			    if { [set $this-color-$plt-$dat] == 0 } {

				set col [expr ([z-$dat $i $j]-[set $this-zmin])*$cOffset]

				if { 1.0 < $col } { set col 1.0 }
				if { $col < 0.0 } { set col 0.0 }
				
				$w cmd plcol1 $col
			    } else {
				$w cmd plcol0 [set $this-color-$plt-$dat]
			    }

			    $w cmd plcol1 $col
			    $w cmd plpoin3 1 xPt yPt zPt 1
			}
		    }
		} else {

# Draw a polygon or a filled polygon.

		    for {set j $inc} {$j < [set $this-jdim-$dat]} {incr j $inc} {
			for {set i $inc} {$i < [set $this-idim-$dat]} {incr i $inc} {

# Create a polygon for this index.
			    set i1 [expr $i-$inc]
			    set j1 [expr $j-$inc]

			    xPoly 0 = [x-$dat $i1 $j1]
			    yPoly 0 = [y-$dat $i1 $j1]
			    zPoly 0 = [z-$dat $i1 $j1]

			    xPoly 1 = [x-$dat $i1 $j]
			    yPoly 1 = [y-$dat $i1 $j]
			    zPoly 1 = [z-$dat $i1 $j]

			    xPoly 2 = [x-$dat $i $j]
			    yPoly 2 = [y-$dat $i $j]
			    zPoly 2 = [z-$dat $i $j]

			    xPoly 3 = [x-$dat $i $j1]
			    yPoly 3 = [y-$dat $i $j1]
			    zPoly 3 = [z-$dat $i $j1]

			    xPoly 4 = [x-$dat $i1 $j1]
			    yPoly 4 = [y-$dat $i1 $j1]
			    zPoly 4 = [z-$dat $i1 $j1]

			    for {set k 0} {$j < 5} {incr j 1} {

				zzPoly 0 $k = [z-$dat $i1 $j1]
				zzPoly 1 $k = [z-$dat $i1 $j]
				zzPoly 2 $k = [z-$dat $i $j]
				zzPoly 3 $k = [z-$dat $i $j1]
				zzPoly 4 $k = [z-$dat $i1 $j1]
			    }

			    set col [expr ([z-$dat $i $j]-[set $this-zmin])*$cOffset]

			    if { 1.0 < $col } { set col 1.0 }
			    if { $col < 0.0 } { set col 0.0 }

			    $w cmd plcol1 $col

			    if { $fill == 1 } {
				$w cmd plline3 5 xPoly yPoly zPoly
			    } elseif { $fill == 2 } {
				$w cmd plpoly3 5 xPoly yPoly zPoly draw
			    } elseif { $fill == 3 } {
				$w cmd plfill3 4 xPoly yPoly zPoly
			    }
			}
		    }
		}
	    }
	}
    }


# Routine for setting colour map1 to rainbow.
    method rainbow_cmap1 {w npts inverse} {

	$w cmd plgcolbg 0 0 0 

# Independent variable of control points.
	matrix p f 12 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }

# Red for control points.
	matrix rf float 12 = { 255.0, 255.0, 255.0, 255.0, 204.0, 102.0, \
				 0.0,   0.0,   0.0,   0.0,   0.0,   0 }
# Green for control points.
	matrix gf float 12 = {   0.0, 102.0, 204.0, 234.0, 255.0, 255.0, \
			       255.0, 255.0, 255.0, 204.0, 102.0,   0 }
# Blue for control points.
	matrix bf float 12 = {   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, \
				 0.0, 102.0, 204.0, 255.0, 255.0, 255 }


# Red for inversed control points.
	matrix ri float 12 = {   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, \
			       102.0, 204.0, 255.0, 255.0, 255.0, 255 }
# Green for inversed control points.
	matrix gi float 12 = {   0.0, 102.0, 204.0, 255.0, 255.0, 255.0, \
			       255.0, 255.0, 234.0, 204.0, 102.0,   0 }
# Blue for inversed control points.
	matrix bi float 12 = { 255.0, 255.0, 255.0, 204.0, 102.0,   0.0, \
				 0.0,   0.0,   0.0,   0.0,   0.0,   0 }

# Make everything 0.0 to 1.0 based.
	for {set i 0} {$i < 12} {incr i} {
	    p $i = [expr ([p $i] / 11.0 )] 
	    rf $i = [expr ([rf $i] / 255.0 )] 
	    gf $i = [expr ([gf $i] / 255.0 )] 
	    bf $i = [expr ([bf $i] / 255.0 )] 
	    ri $i = [expr ([ri $i] / 255.0 )] 
	    gi $i = [expr ([gi $i] / 255.0 )] 
	    bi $i = [expr ([bi $i] / 255.0 )] 
	}

# Integer flag array is zero (no interpolation along far-side of colour figure
	matrix rev i 12 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

# Default number of cmap1 colours
	$w cmd plscmap1n $npts

# Interpolate between control points to set up default cmap1.
	if { $inverse } {
	    $w cmd plscmap1l 1 12 p ri gi bi rev
	} else { 
	    $w cmd plscmap1l 1 12 p rf gf bf rev
	}
    }

















    method x01 {{w loopback}} {
	global xscale yscale xoff yoff

	$w cmd plssub 2 2

	# First plot

	set xscale 6.0
	set yscale 1.0
	set xoff 0.0
	set yoff 0.0

	plot1 $w

	# Second

	set xscale 1.0
	set yscale 0.0014
	set yoff   0.0185

	$w cmd plsyax 5
	plot1 $w

# Third

	plot2 $w

# Fourth

	plot3 $w
	# Restore defaults
	$w cmd plcol0 1
	$w cmd plssub 1 1
	$w cmd pleop
    }

# This is supposed to work just like the plot1() in x01c.c

    method plot1 {w} {
	global xscale yscale xoff yoff

	set npts 60
	matrix x f $npts
	matrix y f $npts

	for {set i 0} {$i < $npts} {incr i} {
	    x $i = [expr $xoff + ($xscale * ($i + 1)) / $npts]
	    y $i = [expr $yoff + $yscale * pow([x $i],2)]
	}

	set xmin [x [expr 0]]
	set xmax [x [expr $npts-1]]
	set ymin [y [expr 0]]
	set ymax [y [expr $npts-1]]

	matrix x1 f 6
	matrix y1 f 6

	for {set i 0} {$i < 6} {incr i} {
	    set j [expr $i*10+3]
	    x1 $i = [x $j]
	    y1 $i = [y $j]
	}

	$w cmd plcol0 1
	$w cmd plenv $xmin $xmax $ymin $ymax 0 0
	$w cmd plcol0 2
	$w cmd pllab "(x)" "(y)" "#frPLplot Example 1 - y=x#u2"

	# plot the data points

	$w cmd plcol0 4
	$w cmd plpoin 6 x1 y1 9

	# draw the line through the data

	$w cmd plcol0 3
	$w cmd plline $npts x y
    }

    # This is supposed to work just like the plot2() in x01c.c

    method plot2 {w} {
	$w cmd plcol0 1
	$w cmd plenv -2 10 -.4 1.2 0 1
	$w cmd plcol0 2
	$w cmd pllab "(x)" "sin(x)/x" "#frPLplot Example 1 - Sinc Function"

	# Fill up the array

	matrix x1 f 101
	matrix y1 f 101

	for {set i 0} {$i < 100} {incr i} {
	    set x [expr ($i - 19.)/6.]
	    x1 $i = $x
	    y1 $i = 1
	    if {$x != 0} { y1 $i = [expr sin($x)/$x] }
	}

	$w cmd plcol0 3
	$w cmd plline 100 x1 y1
    }

    # This is supposed to work just like the plot3() in x01c.c

    method plot3 {w} {

	set pi 3.14159265358979323846
	$w cmd pladv 0
	$w cmd plvsta
	$w cmd plwind 0.0 360.0 -1.2 1.2

	# Draw a box with ticks spaced 60 degrees apart in X, and 0.2 in Y.

	$w cmd plcol0 1
	$w cmd plbox "bcnst" 60.0 2 "bcnstv" 0.2 2

	# Superimpose a dashed line grid, with 1.5 mm marks and spaces. 
	# plstyl expects two integer matrices for mark and space!

	matrix mark i 1
	matrix space i 1

	mark 0 = 1500
	space 0 = 1500
	$w cmd plstyl 1 mark space

	$w cmd plcol0 2
	$w cmd plbox "g" 30.0 0 "g" 0.2 0

	mark 0 = 0
	space 0 = 0
	$w cmd plstyl 0 mark space

	$w cmd plcol0 3
	$w cmd pllab "Angle (degrees)" "sine" "#frPLplot Example 1 - Sine function"

	matrix x f 101
	matrix y f 101

	for {set i 0} {$i < 101} {incr i} {
	    x $i = [expr 3.6 * $i]
	    y $i = [expr sin([x $i] * $pi / 180.0)]
	}

	$w cmd plcol0 4
	$w cmd plline 101 x y
    }

    # Does a series of 3-d plots for a given data set, with different
    # viewing options in each plot.

    # Routine for restoring colour map1 to default.
    # See static void plcmap1_def(void) in plctrl.c for reference.
    method restore_cmap1 {w} {
	# For center control points, pick black or white, whichever is closer to bg 
	set rbg 0
	set gbg 0
	set bbg 0

	# Be careful to pick just short of top or bottom else hue info is lost
	$w cmd plgcolbg rbg gbg bbg
	set vertex [expr ($rbg + $gbg + $bbg)/(3.*255.)]
	if {$vertex < 0.5} {
	    set vertex 0.01
	} else {
	    set vertex 0.99
	}
# Independent variable of control points.
	matrix i f 4 = { 0., 0.45, 0.55, 1.}
# Hue for control points.  Blue-violet to red
	matrix h f 4 = { 260., 260., 0., 0.}
# Lightness ranging from medium to vertex to medium
	matrix l f 4 = { 0.5, $vertex, $vertex, 0.5}
# Saturation is complete for default
	matrix s f 4 = { 1., 1., 1., 1.}
# Integer flag array is zero (no interpolation along far-side of colour figure
	matrix rev i 4 = { 0, 0, 0, 0}
# Default number of cmap1 colours
	$w cmd plscmap1n 128
# Interpolate between control points to set up default cmap1.
	$w cmd plscmap1l 0 4 i h l s rev
    }

    method x08 {{w loopback}} {

	$w cmd plssub 2 2

	matrix opt i 4 = {1, 2, 3, 3}
	matrix alt f 4 = {60.0, 20.0, 60.0, 60.0}
	matrix az  f 4 = {30.0, 60.0, 120.0, 160.0}

	set xpts 35
	set ypts 46
	set n_col 256
	set two_pi [expr 2.0 * 3.14159265358979323846 ]

	matrix x f $xpts
	matrix y f $ypts
	matrix z f $xpts $ypts
	matrix rr i $n_col
	matrix gg i $n_col
	matrix bb i $n_col

	for {set i 0} {$i < $xpts} {incr i} {
	    x $i = [expr ($i - ($xpts/2)) / double($xpts/2) ]
	}

	for {set i 0} {$i < $ypts} {incr i} {
	    y $i = [expr ($i - ($ypts/2)) / double($ypts/2) ]
	}

	for {set i 0} {$i < $xpts} {incr i} {
	    set xx [x $i]
	    for {set j 0} {$j < $ypts} {incr j} {
		set yy [y $j]
		set r [expr sqrt( $xx * $xx + $yy * $yy ) ]

		z $i $j = [expr exp(-$r * $r) * cos( $two_pi * $r ) ]
	    }
	}
	$w cmd pllightsource 1. 1. 1.
	for {set k 0} {$k < $n_col} {incr k} {
	    rr $k = [expr $k]
	    gg $k = [expr $k]
	    bb $k = [expr $k]
	}
	$w cmd  plscmap1 rr gg bb $n_col
	for {set k 0} {$k < 2} {incr k} {
	    for {set ifshade 0} {$ifshade < 2} {incr ifshade} {
		$w cmd pladv 0
		$w cmd plvpor 0.0 1.0 0.0 0.9
		$w cmd plwind -1.0 1.0 -0.9 1.1
		$w cmd plcol0 1
		$w cmd plw3d 1.0 1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 [alt $k] [az $k]
		$w cmd plbox3 "bnstu" "x axis" 0.0 0 \
		    "bnstu" "y axis" 0.0 0 \
		    "bcdmnstuv" "z axis" 0.0 0
		$w cmd plcol0 2
		if {$ifshade == 1} {
		    $w cmd plotsh3d x y z 0
		} else {
		    $w cmd plot3d x y z [opt $k] 1
		}
		$w cmd plcol0 3
		set title [format "#frPLplot Example 8 - Alt=%.0f, Az=%.0f, Opt=%d" \
			       [alt $k] [az $k] [opt $k] ]
		$w cmd plmtex "t" 1.0 0.5 0.5 $title
	    }
	}

	# Restore defaults
	$w cmd plcol0 1
	restore_cmap1 $w
    }

    method x16 {{w loopback}} {

	$w cmd plssub 2 2

	set ns 20
	set nx 35
	set ny 46

	set pi 3.14159265358979323846

	set sh_cmap 1
	set min_color 1; set min_width 0; set max_color 0; set max_width 0

	matrix clevel f $ns
	matrix xg1 f $nx
	matrix yg1 f $ny
	matrix xg2 f $nx $ny
	matrix yg2 f $nx $ny
	matrix zz f $nx $ny
	matrix ww f $nx $ny

	# Set up data array

	for {set i 0} {$i < $nx} {incr i} {
	    set x [expr ($i - ($nx/2.)) / ($nx/2.)]
	    for {set j 0} {$j < $ny} {incr j} {
		set y [expr ($j - .5 * $ny) / (.5 * $ny) - 1.]

		zz $i $j = [expr -sin(7.*$x) * cos(7.*$y) + $x*$x - $y*$y ]
		ww $i $j = [expr -cos(7.*$x) * sin(7.*$y) + 2 * $x * $y ]
	    }
	}

	set zmin [zz 0 0]
	set zmax $zmin
	for {set i 0} {$i < $nx} {incr i} {
	    for {set j 0} {$j < $ny} {incr j} {
		if {[zz $i $j] < $zmin} { set zmin [zz $i $j] }
		if {[zz $i $j] > $zmax} { set zmax [zz $i $j] }
	    }
	}

	for {set i 0} {$i < $ns} {incr i} {
	    clevel $i = [expr $zmin + ($zmax - $zmin) * ($i + .5) / $ns.]
	}

	# Build the 1-d coord arrays.

	set distort .4

	for {set i 0} {$i < $nx} {incr i} {
	    set xx [expr -1. + $i * ( 2. / ($nx-1.) )]
	    xg1 $i = [expr $xx + $distort * cos( .5 * $pi * $xx ) ]
	}

	for {set j 0} {$j < $ny} {incr j} {
	    set yy [expr -1. + $j * ( 2. / ($ny-1.) )]
	    yg1 $j = [expr $yy - $distort * cos( .5 * $pi * $yy ) ]
	}

	# Build the 2-d coord arrays.

	for {set i 0} {$i < $nx} {incr i} {
	    set xx [expr -1. + $i * ( 2. / ($nx-1.) )]
	    for {set j 0} {$j < $ny} {incr j} {
		set yy [expr -1. + $j * ( 2. / ($ny-1.) )]

		set argx [expr .5 * $pi * $xx]
		set argy [expr .5 * $pi * $yy]

		xg2 $i $j = [expr $xx + $distort * cos($argx) * cos($argy) ]
		yg2 $i $j = [expr $yy - $distort * cos($argx) * cos($argy) ]
	    }
	}

	# Plot using identity transform

	$w cmd pladv 0
	$w cmd plvpor 0.1 0.9 0.1 0.9
	$w cmd plwind -1.0 1.0 -1.0 1.0

	for {set i 0} {$i < $ns} {incr i} {
	    set shade_min [expr $zmin + ($zmax - $zmin) * $i / $ns.]
	    set shade_max [expr $zmin + ($zmax - $zmin) * ($i +1.) / $ns.]
	    set sh_color [expr $i. / ($ns-1.)]
	    set sh_width 2
	    $w cmd plpsty 0

	    #	plshade(z, nx, ny, NULL, -1., 1., -1., 1., 
	    #		shade_min, shade_max, 
	    #		sh_cmap, sh_color, sh_width,
	    #		min_color, min_width, max_color, max_width,
	    #		plfill, 1, NULL, NULL);
	    $w cmd plshade zz -1. 1. -1. 1. \
		$shade_min $shade_max $sh_cmap $sh_color $sh_width \
		$min_color $min_width $max_color $max_width \
		1
	}

	$w cmd plcol0 1
	$w cmd plbox "bcnst" 0.0 0 "bcnstv" 0.0 0
	$w cmd plcol0 2

	#    plcont(w, nx, ny, 1, nx, 1, ny, clevel, ns, mypltr, NULL);

	$w cmd pllab "distance" "altitude" "Bogon density"

	# Plot using 1d coordinate transform
	
	$w cmd pladv 0
	$w cmd plvpor 0.1 0.9 0.1 0.9
	$w cmd plwind -1.0 1.0 -1.0 1.0

	for {set i 0} {$i < $ns} {incr i} {
	    set shade_min [expr $zmin + ($zmax - $zmin) * $i / $ns.]
	    set shade_max [expr $zmin + ($zmax - $zmin) * ($i +1.) / $ns.]
	    set sh_color [expr $i. / ($ns-1.)]
	    set sh_width 2
	    $w cmd plpsty 0

	    #	plshade(z, nx, ny, NULL, -1., 1., -1., 1., 
	    #		shade_min, shade_max, 
	    #		sh_cmap, sh_color, sh_width,
	    #		min_color, min_width, max_color, max_width,
	    #		plfill, 1, pltr1, (void *) &cgrid1);

	    $w cmd plshade zz -1. 1. -1. 1. \
		$shade_min $shade_max $sh_cmap $sh_color $sh_width \
		$min_color $min_width $max_color $max_width \
		1 pltr1 xg1 yg1
	}

	$w cmd plcol0 1
	$w cmd plbox "bcnst" 0.0 0 "bcnstv" 0.0 0
	$w cmd plcol0 2

	#    plcont(w, nx, ny, 1, nx, 1, ny, clevel, ns, pltr1, (void *) &cgrid1);

	$w cmd pllab "distance" "altitude" "Bogon density"

	# Plot using 2d coordinate transform

	$w cmd pladv 0
	$w cmd plvpor 0.1 0.9 0.1 0.9
	$w cmd plwind -1.0 1.0 -1.0 1.0

	for {set i 0} {$i < $ns} {incr i} {
	    set shade_min [expr $zmin + ($zmax - $zmin) * $i / $ns.]
	    set shade_max [expr $zmin + ($zmax - $zmin) * ($i +1.) / $ns.]
	    set sh_color [expr $i. / ($ns-1.)]
	    set sh_width 2
	    $w cmd plpsty 0

	    $w cmd plshade zz -1. 1. -1. 1. \
		$shade_min $shade_max $sh_cmap $sh_color $sh_width \
		$min_color $min_width $max_color $max_width \
		0 pltr2 xg2 yg2
	}

	$w cmd plcol0 1
	$w cmd plbox "bcnst" 0.0 0 "bcnstv" 0.0 0
	$w cmd plcol0 2
	#    plcont(w, nx, ny, 1, nx, 1, ny, clevel, ns, pltr2, (void *) &cgrid2);
	$w cmd plcont ww clevel pltr2 xg2 yg2

	$w cmd pllab "distance" "altitude" "Bogon density, with streamlines"

	# Do it again, but show wrapping support.

	$w cmd pladv 0
	$w cmd plvpor 0.1 0.9 0.1 0.9
	$w cmd plwind -1.0 1.0 -1.0 1.0

	# Hold perimeter
	matrix px f 100; matrix py f 100

	for {set i 0} {$i < 100} {incr i} {
	    set t [expr 2. * $pi * $i / 99.]
	    px $i = [expr cos($t)]
	    py $i = [expr sin($t)]
	}
	# We won't draw it now b/c it should be drawn /after/ the plshade stuff.

	# Now build the new coordinate matricies.

	matrix xg f $nx $ny
	matrix yg f $nx $ny
	matrix z  f $nx $ny

	for {set i 0} {$i < $nx} {incr i} {
	    set r [expr $i / ($nx - 1.)]
	    for {set j 0} {$j < $ny} {incr j} {
		set t [expr 2. * $pi * $j / ($ny - 1.)]

		xg $i $j = [expr $r * cos($t)]
		yg $i $j = [expr $r * sin($t)]

		z $i $j = [expr exp(-$r*$r) * cos(5.*$t) * cos(5.*$pi*$r) ]
	    }
	}

	# Need a new clevel to go allong with the new data set.

	set zmin [z 0 0]
	set zmax $zmin
	for {set i 0} {$i < $nx} {incr i} {
	    for {set j 0} {$j < $ny} {incr j} {
		if {[z $i $j] < $zmin} { set zmin [z $i $j] }
		if {[z $i $j] > $zmax} { set zmax [z $i $j] }
	    }
	}

	for {set i 0} {$i < $ns} {incr i} {
	    clevel $i = [expr $zmin + ($zmax - $zmin) * ($i + .5) / $ns.]
	}

	# Now we can shade the interior region.

	for {set i 0} {$i < $ns} {incr i} {
	    set shade_min [expr $zmin + ($zmax - $zmin) * $i / $ns.]
	    set shade_max [expr $zmin + ($zmax - $zmin) * ($i +1.) / $ns.]
	    set sh_color [expr $i. / ($ns-1.)]
	    set sh_width 2
	    $w cmd plpsty 0

	    $w cmd plshade z -1. 1. -1. 1. \
		$shade_min $shade_max $sh_cmap $sh_color $sh_width \
		$min_color $min_width $max_color $max_width \
		0 pltr2 xg yg 2
	}

	# Now we can draw the perimeter.
	$w cmd plcol0 1
	$w cmd plline 100 px py

	# And label the plot.
	$w cmd plcol0 2
	$w cmd pllab "" "" "Tokamak Bogon Instability"
    }
}
