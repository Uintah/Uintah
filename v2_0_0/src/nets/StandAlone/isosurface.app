# SCI Network 1.0

package require Iwidgets 3.0

::netedit dontschedule

global notes
set notes ""

# global array indexed by module name to keep track of modules
global mods

set m0 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 111 44]
set mods(FieldReader) $m0

set m1 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 111 283]
set mods(Isosurface) $m1

set m2 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 300 46]
set mods(GenStandardColorMaps) $m2

set m3 [addModuleAtPosition "SCIRun" "Render" "Viewer" 129 384]
set mods(Viewer) $m3

set m4 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 300 185]
set mods(RescaleColorMap) $m4


addConnection $m2 0 $m4 0
addConnection $m0 0 $m4 1
addConnection $m4 0 $m1 1
addConnection $m0 0 $m1 0
addConnection $m1 1 $m3 0

set $m0-notes {}
set $m0-show_status {1}
set $m0-filename {}
set $m1-notes {}
set $m1-show_status {1}
set $m1-isoval {0.0}
set $m1-isoval-min {0.0}
set $m1-isoval-max {100.0}
set $m1-isoval-typed {0}
set $m1-isoval-quantity {1}
set $m1-quantity-range {colormap}
set $m1-quantity-min {0}
set $m1-quantity-max {100}
set $m1-isoval-list {0.0 1.0 2.0 3.0}
set $m1-extract-from-new-field {1}
set $m1-algorithm {0}
set $m1-build_trisurf {0}
set $m1-np {1}
set $m1-active-isoval-selection-tab {0}
set $m1-active_tab {MC}
set $m1-update_type {on release}
set $m1-color-r {0.4}
set $m1-color-g {0.2}
set $m1-color-b {0.9}
set $m2-notes {}
set $m2-show_status {1}
set $m2-tcl_status {Calling GenStandardColorMaps!}
set $m2-positionList {}
set $m2-nodeList {}
set $m2-width {1}
set $m2-height {1}
set $m2-mapType {3}
set $m2-minRes {12}
set $m2-resolution {256}
set $m2-realres {256}
set $m2-gamma {0}
set $m3-notes {}
set $m3-show_status {1}
set $m4-notes {}
set $m4-show_status {1}
set $m4-isFixed {0}
set $m4-min {-5.25315}
set $m4-max {4.31926}
set $m4-makeSymmetric {0}

::netedit scheduleok


#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

class IsoApp {

    method modname {} {
	return "IsoApp"
    }

    constructor {} {
	global mods

	toplevel .standalone
	wm title .standalone "BioIso"	 
	set win .standalone

	set notebook_width 270
	set notebook_height 315

	set iso_alg 1

	set viewer_width 640
	set viewer_height 512
    
	set process_width 220
	set process_height $viewer_height

	set vis_width 300
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

	set error_module ""
        set current_step "Loading"
	set current_data ""
	set data_frame1 ""
	set data_frame2 ""

        set data_loaded 0

	set vis_tab "Data Vis"

        set data_label ""

	# block FieldReader to Isosurface
	# block_connection $mods(FieldReader) 0 $mods(Isosurface) 0 "yellow"

	# block FieldReader to RescaleColorMap
	# block_connection $mods(FieldReader) 0 $mods(RescaleColorMap) 1 "yellow"

	# block RescaleColorMap to Isosurface
	# block_connection $mods(RescaleColorMap) 0 $mods(Isosurface) 1 "purple"

    }

    destructor {
	destroy $this
    }

    method build_app {} {
	global mods

	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer


	set att_msg "Detach from Viewer"
	set det_msg " Attach to Viewer "


	### Processing Part
	#########################
	### Create Detached Processing Part
	toplevel $win.detachedP
	frame $win.detachedP.f -relief flat
	pack $win.detachedP.f -side left -anchor n -fill both -expand 1

	wm title $win.detachedP "Processing Window"
	
	wm sizefrom $win.detachedP user
	wm positionfrom $win.detachedP user

	wm withdraw $win.detachedP


	### Create Attached Processing Part
	frame $win.attachedP 
	frame $win.attachedP.f -relief flat 
	pack $win.attachedP.f -side top -anchor n -fill both -expand 1

	set IsPAttached 1

	# set frame data members
	set detachedPFr $win.detachedP
	set attachedPFr $win.attachedP

	init_Pframe $detachedPFr.f $det_msg 0
	init_Pframe $attachedPFr.f $att_msg 1

	# create detached width and heigh
	append geomP $process_width x $process_height
	wm geometry $detachedPFr $geomP


	### Vis Part
	#####################
	### Create a Detached Vis Part
	toplevel $win.detachedV
	frame $win.detachedV.f -relief flat
	pack $win.detachedV.f -side left -anchor n

	wm title $win.detachedV "Visualization Window"

	wm sizefrom $win.detachedV user
	wm positionfrom $win.detachedV user
	
	wm withdraw $win.detachedV

	### Create Attached Vis Part
	frame $win.attachedV
	frame $win.attachedV.f -relief flat
	pack $win.attachedV.f -side left -anchor n -fill both

	set IsVAttached 1

	# set frame data members
	set detachedVFr $win.detachedV
	set attachedVFr $win.attachedV
	
	init_Vframe $detachedVFr.f $det_msg 0
	init_Vframe $attachedVFr.f $att_msg 1


	# pack 3 frames
	pack $attachedPFr $win.viewer $attachedVFr -side left -anchor n -fill both -expand 1

	set total_width [expr [expr $process_width + $viewer_width] + $vis_width]

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]

	append geom $total_width x $viewer_height + $pos_x + 0
	wm geometry .standalone $geom
	update	
    }


    method init_Pframe { m msg case } {

	if { [winfo exists $m] } {
	    ### Menu
	    frame $m.main_menu -relief raised -borderwidth 3
	    pack $m.main_menu -fill x -anchor nw
	    
	    menubutton $m.main_menu.file -text "File" -underline 0 \
		-menu $m.main_menu.file.menu
	    
	    menu $m.main_menu.file.menu -tearoff false
	    
	    $m.main_menu.file.menu add command -label "Save Ctr+S" \
		-underline 0 -command "$this save_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Load  Ctr+O" \
		-underline 0 -command "$this load_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Quit   Ctr+Q" \
		-underline 0 -command "$this exit_app" -state active
	    
	    pack $m.main_menu.file -side left
	    
	    menubutton $m.main_menu.help -text "Help" -underline 0 \
		-menu $m.main_menu.help.menu
	    
	    menu $m.main_menu.help.menu -tearoff false
	    
	    $m.main_menu.help.menu add command -label "View Help" \
		-underline 0 -command "$this show_help" -state active
	    
	    pack $m.main_menu.help -side left
	    
	    tk_menuBar $m.main_menu $win.main_menu.file $win.main_menu.help
	    
	    ### Processing Steps
	    #####################
	    iwidgets::labeledframe $m.p \
		-labelpos n -labeltext "Processing Steps" 
		# -background "MistyRose"
	    pack $m.p -side left -fill both -anchor n -expand 1
	    
	    set process [$m.p childsite]
	    
	    ### Data Section
	    iwidgets::labeledframe $process.data \
		-labelpos nw -labeltext "Data" 
	    pack $process.data -side top -fill both -anchor n
	    
	    set data_section [$process.data childsite]
	    
	    message $data_section.message -width 200 \
		-text "Please load a dataset to isosurface."
	    pack $data_section.message -side top  -anchor n
	    button $data_section.load -text "Load" \
	        -command "$this popup_load_data"  \
		-width 15 
	    pack $data_section.load -side top -padx 2 -pady 5  -ipadx 3 -ipady 3 -anchor n
	    
	    # Tooltip $data_section.load "Load data set"
	
	    ### Progress
	    iwidgets::labeledframe $process.progress \
		-labelpos nw -labeltext "Progress" 
	    pack $process.progress -side bottom -anchor s -fill both
	    
	    set progress_section [$process.progress childsite]
	    iwidgets::feedback $progress_section.fb -labeltext "$current_step..." \
		-labelpos nw \
		-steps 30 -barcolor Green \
		
	    pack $progress_section.fb -side top -padx 2 -pady 2 -anchor nw -fill both
	    if {$case == 0} {
	        set standalone_progress1 $progress_section.fb
                bind $standalone_progress1.lwchildsite.trough <Button> { app display_module_error }
                bind $standalone_progress1.lwchildsite.trough.bar <Button> { app display_module_error }

	        # Tooltip $standalone_progress1.lwchildsite.trough "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress1.lwchildsite.trough.bar "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress1.label "Indicates current step"
            } else {
	        set standalone_progress2 $progress_section.fb
                bind $standalone_progress2.lwchildsite.trough <Button> { app display_module_error }
                bind $standalone_progress2.lwchildsite.trough.bar <Button> { app display_module_error }

	        # Tooltip $standalone_progress2.lwchildsite.trough "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress2.lwchildsite.trough.bar "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress2.label "Indicates current step"

            }
	
	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<25} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_P_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
                # Tooltip $m.d.cut$i "Click to $msg"
            }



	}
	    
    }

    method init_Vframe { m msg case} {
	global mods
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization" 
	        # -background "LightSteelBlue3"
	    pack $m.vis -side right -anchor n -fill both -expand 1
	    
	    set vis [$m.vis childsite]

            if {$case == 0} {
               set vis1 $vis
            } else {
               set vis2 $vis
            }
         

	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height 490 -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	    ### Data Vis Tab
	    set page [$vis.tnb add -label "Data Vis" -command "set vis_tab \"Data Vis\""]
	    iwidgets::scrolledlistbox $page.data -labeltext "Loaded Data" \
		-vscrollmode dynamic -hscrollmode dynamic \
		-selectmode single \
		-height 0.9i \
		-width $notebook_width \
		-labelpos nw -selectioncommand "$this data_selected"
	    
	    pack $page.data -padx 4 -pady 4 -anchor n
	    
	    if {$case == 0} {
		# detached case
		set data_listbox_Det $page.data
	    } else {
		# attached case
		set data_listbox_Att $page.data
	    }
	    
	    
	    
	    ### Data Info
	    frame $page.f -relief groove -borderwidth 2
	    pack $page.f -side top -anchor n -fill both -expand 1

	    if {$case == 0} {
              set data_frame1 $page.f
            } else {
              set data_frame2 $page.f
            }
	
            label $page.f.datasetl -text "Data: (None)" 
            pack $page.f.datasetl -side top -anchor nw	

            if {$case == 0} {
              set data_label1 $page.f.datasetl
            } else {
              set data_label2 $page.f.datasetl
            }

            add_isosurface_section $page.f $case
	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis


	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side right -anchor w
            for {set i 0} {$i<27} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_V_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
                # Tooltip $m.d.cut$i "Click to $msg"
            }
	}
    }

    method create_viewer_tab { vis } {
	global mods
	set page [$vis.tnb add -label "Global Options" -command "set vis_tab \"Global Options\""]
	
	iwidgets::labeledframe $page.viewer_opts \
	    -labelpos nw -labeltext "Global Render Options"
	
	pack $page.viewer_opts -side top -anchor n -fill both -expand 1
	
	set view_opts [$page.viewer_opts childsite]
	
	frame $view_opts.eframe -relief flat
	pack $view_opts.eframe -side top -padx 4 -pady 4
	
	frame $view_opts.eframe.a -relief groove -borderwidth 2
	pack $view_opts.eframe.a -side left 
	
	
	checkbutton $view_opts.eframe.a.light -text "Lighting" \
	    -variable $mods(Viewer)-ViewWindow_0-global-light \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.fog -text "Fog" \
	    -variable $mods(Viewer)-ViewWindow_0-global-fog \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.bbox -text "BBox" \
	    -variable $mods(Viewer)-ViewWindow_0-global-debug \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	pack $view_opts.eframe.a.light $view_opts.eframe.a.fog \
	    $view_opts.eframe.a.bbox  \
	    -side top -anchor w -padx 4 -pady 4
	
	frame $view_opts.buttons -relief flat
	pack $view_opts.buttons -side top -anchor n -padx 4 -pady 4
	
	frame $view_opts.buttons.v1
	pack $view_opts.buttons.v1 -side left -anchor nw
	
	
	button $view_opts.buttons.v1.autoview -text "Autoview" \
	    -command "$mods(Viewer)-ViewWindow_0-c autoview" \
	    -width 12 -padx 3 -pady 3
	
	# Tooltip $view_opts.buttons.v1.autoview "Restore display to\ndefault view"

	pack $view_opts.buttons.v1.autoview -side top -padx 3 -pady 3 \
	    -anchor n -fill x
	
	
	frame $view_opts.buttons.v1.views
	pack $view_opts.buttons.v1.views -side top -anchor nw -fill x -expand 1
	
	menubutton $view_opts.buttons.v1.views.def -text "Views" \
	    -menu $view_opts.buttons.v1.views.def.m -relief raised \
	    -padx 3 -pady 3  -width 12

	# Tooltip $view_opts.buttons.v1.views.def "Standard viewing angles\nand orientations"
	
	menu $view_opts.buttons.v1.views.def.m
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +X Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posx
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +Y Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posy
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +Z Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posz
	$view_opts.buttons.v1.views.def.m add separator
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -X Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negx
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -Y Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negy
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -Z Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negz
	
	pack $view_opts.buttons.v1.views.def -side left -pady 3 -padx 3 -fill x
	
	menu $view_opts.buttons.v1.views.def.m.posx
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.posy
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.posz
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negx
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negy
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negz
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	
	frame $view_opts.buttons.v2 
	pack $view_opts.buttons.v2 -side left -anchor nw
	
	button $view_opts.buttons.v2.sethome -text "Set Home View" \
            -padx 3 -pady 3 \
	    -command "$mods(Viewer)-ViewWindow_0-c sethome"

	# Tooltip $view_opts.buttons.v2.sethome "Save current view\nto return to later by\nclicking Go Home button"
	
	button $view_opts.buttons.v2.gohome -text "Go Home" \
	    -command "$mods(Viewer)-ViewWindow_0-c gohome" \
	    -padx 3 -pady 3
        # # Tooltip $view_opts.buttons.v2.gohome "Restore current\nhome view"
	
	pack $view_opts.buttons.v2.sethome $view_opts.buttons.v2.gohome \
	    -side top -padx 2 -pady 2 -anchor ne -fill x
	
	$vis.tnb view "Data Vis"
	
    }


    method switch_P_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

    	set x [winfo x $win]
	set y [expr [winfo y $win] - 20]

	if { $IsPAttached } {	    
	    pack forget $attachedPFr
	    set new_width [expr $c_width - $process_width]
	    append geom1 $new_width x $c_height + [expr $x+$process_width] + $y
            wm geometry $win $geom1 
	    append geom2 $process_width x $c_height + [expr $x-20] + $y
	    wm geometry $detachedPFr $geom2
	    wm deiconify $detachedPFr
	    set IsPAttached 0
	} else {
	    wm withdraw $detachedPFr
	    pack $attachedPFr -anchor n -side left -before $win.viewer \
	      -expand 1 -fill both
	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height + [expr $x - $process_width] + $y
	    wm geometry $win $geom
	    set IsPAttached 1
	}
	update
    }

    method switch_V_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

      	set x [winfo x $win]
	set y [expr [winfo y $win] - 20]


	if { $IsVAttached } {
	    # select data in detached data list box
	    if {[$data_listbox_Att curselection] != ""} {
		$data_listbox_Det selection set [$data_listbox_Att curselection] [$data_listbox_Att curselection]
	    }
	    pack forget $attachedVFr
	    set new_width [expr $c_width - $vis_width]
	    append geom1 $new_width x $c_height
            wm geometry $win $geom1
	    set move [expr $c_width - $vis_width]
	    append geom2 $vis_width x $c_height + [expr $x + $move + 20] + $y
	    wm geometry $detachedVFr $geom2
	    wm deiconify $detachedVFr
	    set IsVAttached 0
	} else {
	    if {[$data_listbox_Det curselection] != ""} {
	    $data_listbox_Att selection set [$data_listbox_Det curselection] [$data_listbox_Det curselection]
	    }

	    wm withdraw $detachedVFr
	    pack $attachedVFr -anchor n -side left -after $win.viewer \
	      -expand 1 -fill both
	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
	update
    }

    method popup_load_data {} {
	# Bring up FieldReader UI
	global mods
	global $mods(FieldReader)-filename
	$mods(FieldReader) ui
	
	tkwait window .ui$mods(FieldReader)

	update idletasks

	load_data [set $mods(FieldReader)-filename]
    }

    method load_data { file } {
        if {$data_loaded} {
           save_isosurface_state
        }

	global mods
	global $mods(FieldReader)-filename
        global $mods(Isosurface)-isoval
        global $mods(Isosurface)-isoval-min
        global $mods(Isosurface)-isoval-max
        global $mods(Isosurface)-isoval-list
        global $mods(Isosurface)-isoval-quantity
        global $mods(Isosurface)-quantity-min
        global $mods(Isosurface)-quantity-max


	# set the FieldReader filename if it hasn't been set
	if {[set $mods(FieldReader)-filename] == ""} {
	    set $mods(FieldReader)-filename $file
	    $mods(FieldReader)-c needexecute
	}
	
	# get the filename and stick into an array formated 
	# data(file) = full path
	set name [get_file_name $file]
	set data($name) $file
        set current_data $name

        # initialize isosurface vars
        set isovals($name) [set $mods(Isosurface)-isoval]
        set mins($name) [set $mods(Isosurface)-isoval-min]
        set maxs($name) [set $mods(Isosurface)-isoval-max]
        set $mods(Isosurface)-isoval-quantity 1
        set quantity($name) 1
        set quantity_min($name) [set $mods(Isosurface)-isoval-min]
        set quantity_max($name) [set $mods(Isosurface)-isoval-max]
        set $mods(Isosurface)-quantity-min [set $mods(Isosurface)-isoval-min]
        set $mods(Isosurface)-quantity-max [set $mods(Isosurface)-isoval-max]
        set isoval_list($name) [set $mods(Isosurface)-isoval-list]
        set $mods(Isosurface)-isoval-list {0.0 1.0 2.0 3.0}
        set iso_page($name) "Slider"
	
	# add the data to the listbox
        add_data_to_listbox $name

        # set label
        $data_label1 configure -text "Data: $current_data"
        $data_label2 configure -text "Data: $current_data"

	# reset progress
	$standalone_progress1 reset
	$standalone_progress2 reset

	set current_step "Isosurfacing"
	$standalone_progress1 configure -labeltext "$current_step..."
	$standalone_progress2 configure -labeltext "$current_step..."

        set data_loaded 1
    }

    method add_data_to_listbox {name} {
	if {[array size data] == 1} then {
	    # first data set should replace "none" in listbox and be selected
	    $data_listbox_Att delete 0
	    $data_listbox_Att insert 0 $name

	    $data_listbox_Det delete 0
	    $data_listbox_Det insert 0 $name

	    if {$IsVAttached} {
		$data_listbox_Att selection clear 0 end
		$data_listbox_Att selection set 0 0
		$data_listbox_Att see 0
	    } else {
		$data_listbox_Det selection clear 0 end
		$data_listbox_Det selection set 0 0
		$data_listbox_Det see 0
	    }
	} else {
	    # otherwise add to bottom and select
	    $data_listbox_Att insert end $name
	    $data_listbox_Det insert end $name

	    if {$IsVAttached} {
		$data_listbox_Att selection clear 0 end
		$data_listbox_Att selection set end end
		$data_listbox_Att see end
	    } else {
		$data_listbox_Det selection clear 0 end
		$data_listbox_Det selection set end end
		$data_listbox_Det see end
	    }
	}
    }

    method data_selected {} {
	global mods
	global $mods(FieldReader)-filename

        # save out old isosurface settings
        save_isosurface_state
        
	set has_changed 0
	if {$IsVAttached} {
            if {$current_data != [$data_listbox_Att getcurselection]} {
                set has_changed 1
            }
	    set current_data [$data_listbox_Att getcurselection]
	} else {
            if {$current_data != [$data_listbox_Det getcurselection]} {
                set has_changed 1
            }
	    set current_data [$data_listbox_Det getcurselection]
	}

	if {[info exists data($current_data)] && $has_changed} {
            $data_label1 configure -text "Data: $current_data"
            $data_label2 configure -text "Data: $current_data"

	    # data selected - update FieldReader and execute
	    set $mods(FieldReader)-filename $data($current_data)

	    # configure isosurface widgets specific to this dataset

	    restore_isosurface_state

            $standalone_progress1 reset
            $standalone_progress2 reset

	    $mods(FieldReader)-c needexecute

	} 
   }

    method save_session {} {
	global mods

	set types {
	    {{App Settings} {.set} }
	    {{Other} { * } }
	} 
	set savefile [ tk_getSaveFile -defaultextension {.set} \
				   -filetypes $types ]
	if { $savefile != "" } {
	    set fileid [open $savefile w]

            puts $fileid "set app_name \"BioIso\""
	    puts $fileid "\nglobal mods\n"

	    puts $fileid "\#Save out data sets and isosurface information"
	    set searchId [array startsearch data]

            while { [array anymore data $searchId] } {
                set temp [array nextelement data $searchId]
	        puts $fileid "set data($temp) $data($temp)"
                puts $fileid "set isovals($temp) $isovals($temp)"
                puts $fileid "set mins($temp) $mins($temp)"
                puts $fileid "set maxs($temp) $maxs($temp)"
                puts $fileid "set quantity($temp) $quantity($temp)"
                puts $fileid "set quantity_min($temp) $quantity_min($temp)"
                puts $fileid "set quantity_max($temp) $quantity_max($temp)"
                puts $fileid "set isoval_list($temp) {$isoval_list($temp)}\n\n"
            }

            array donesearch data $searchId

            puts $fileid "# current data set\nset current {$current_data}\n"

            global $mods(Isosurface)-active-isoval-selection-tab

            puts $fileid "global \$mods(Isosurface)-active-isoval-selection-tab\nset \$mods(Isosurface)-active-isoval-selection-tab [set $mods(Isosurface)-active-isoval-selection-tab]"
	    
	    # Save out global rendering properties
	    global $mods(Viewer)-ViewWindow_0-global-light
	    global $mods(Viewer)-ViewWindow_0-global-fog
	    global $mods(Viewer)-ViewWindow_0-global-debug

  	    puts $fileid "\n\#Global Rendering Options"
  	    puts $fileid "global \$mods(Viewer)-ViewWindow_0-global-light\nset \$mods(Viewer)-ViewWindow_0-global-light [set $mods(Viewer)-ViewWindow_0-global-light]"
  	    puts $fileid "global \$mods(Viewer)-ViewWindow_0-global-fog\nset \$mods(Viewer)-ViewWindow_0-global-fog [set $mods(Viewer)-ViewWindow_0-global-fog]"
  	    puts $fileid "global \$mods(Viewer)-ViewWindow_0-global-debug\nset \$mods(Viewer)-ViewWindow_0-global-debug [set $mods(Viewer)-ViewWindow_0-global-debug]"
	    

            puts $fileid "\n\# Visualization tab"
            puts $fileid "set visualization_tab {$vis_tab}"

            puts $fileid "\n\# Save Viewer Settings"
		
            set result [$mods(Viewer)-ViewWindow_0-c autoview]
	    close $fileid

	}

    }

    method save_advanced { id vars } {
	# make globals accessible
	foreach g [info globals] {
	   global $g
	}

	
	foreach v [info vars $vars*] {
	  puts $id "set $v \{[set $v]\}"
	}
    }
    
    method load_session {} {
	global mods
	set types {
	    {{App Settings} {.set} }
	    {{Other} { * }}
	}

	set file [tk_getOpenFile -filetypes $types]
	if {$file != ""} {
   	    source $file

            # add each data set to listbox
	    set searchId [array startsearch data]
            while { [array anymore data $searchId] } {
                 add_data_to_listbox [array nextelement data $searchId]
            }
            array donesearch data $searchId
        
            # set appropriate isosurface values 
            set current_data $current
            restore_isosurface_state

            # set the appropriate dataset to be executed on
	    global $mods(FieldReader)-filename
            set $mods(FieldReader)-filename $data($current_data)

            # set iso tab
            global $mods(Isosurface)-active-isoval-selection-tab
            if {[set $mods(Isosurface)-active-isoval-selection-tab]==0} {
	        $isosurface1.tnb view "Slider"
	        $isosurface2.tnb view "Slider"
            } elseif {[set $mods(Isosurface)-active-isoval-selection-tab]==1} {
	        $isosurface1.tnb view "Quantity"
	        $isosurface2.tnb view "Quantity"
            } elseif {[set $mods(Isosurface)-active-isoval-selection-tab]==2} {
	        $isosurface1.tnb view "List"
	        $isosurface2.tnb view "List"
            }

	    # bring appropriate vis tab forward
	    set vis_tab $visualization_tab
            $vis1.tnb view "$vis_tab"
            $vis2.tnb view "$vis_tab"

            # execute
            $mods(Isosurface)-c needexecute
	}	
    }
        
    method exit_app {} {
	netedit quit
    }

    method show_help {} {
	puts "NEED TO IMPLEMENT SHOW HELP"
    }

    method update_isosurface { isoval } {
	if {$data_loaded} {
   	   global mods
	   set isovals($current_data) $isoval

	   # execute with the given value
	   [$mods(Isosurface) get_this_c] needexecute
        }

    }
	
    method print_iso {} {
	parray $isosurface1
	parray $isosurface2
    }


    method save_isosurface_state {} {
        global mods

        global $mods(Isosurface)-isoval
        global $mods(Isosurface)-isoval-min
        global $mods(Isosurface)-isoval-max
        global $mods(Isosurface)-isoval-quantity
        global $mods(Isosurface)-quantity-min
        global $mods(Isosurface)-quantity-max
        global $mods(Isosurface)-isoval-list

        set isovals($current_data) [set $mods(Isosurface)-isoval]
        set mins($current_data) [set $mods(Isosurface)-isoval-min]
        set maxs($current_data) [set $mods(Isosurface)-isoval-max]

        set quantity($current_data) [set $mods(Isosurface)-isoval-quantity]
        set quantity_min($current_data) [set $mods(Isosurface)-quantity-min]
        set quantity_max($current_data) [set $mods(Isosurface)-quantity-max]

        set isoval_list($current_data) [set $mods(Isosurface)-isoval-list]

    }


    method restore_isosurface_state {} {
        global mods

        set temp1 $isosurface1.tnb.canvas.notebook.cs
        set temp2 $isosurface2.tnb.canvas.notebook.cs

        # Slider
        $temp1.page1.cs.isoval configure -from $mins($current_data) -to $maxs($current_data)

	$temp2.page1.cs.isoval configure -from $mins($current_data) -to $maxs($current_data)

        global $mods(Isosurface)-isoval
	set $mods(Isosurface)-isoval $isovals($current_data)

        # Quantity
        global $mods(Isosurface)-isoval-quantity
        global $mods(Isosurface)-quantity-min
        global $mods(Isosurface)-quantity-max
        set $mods(Isosurface)-isoval-quantity $quantity($current_data)
        set $mods(Isosurface)-quantity-min $quantity_min($current_data)
        set $mods(Isosurface)-quantity-max $quantity_max($current_data)

        global $mods(Isosurface)-isoval-quantity
        set $mods(Isosurface)-isoval-quantity $quantity($current_data)

        # List
	global $mods(Isosurface)-isoval-list
	set $mods(Isosurface)-isoval-list $isoval_list($current_data)

    }

    method set_isosurface_min_max {} {
	global mods
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

	set min [set $mods(Isosurface)-isoval-min]
	set max [set $mods(Isosurface)-isoval-max]

	if {[winfo exists $isosurface1($current_data)]} {
          # set to and from of isosurface slider tab
	  # $isosurface1($current_data).tnb.canvas.notebook.cs.page1.cs.isoval configure -from $min -to $max
	  # $isosurface2($current_data).tnb.canvas.notebook.cs.page1.cs.isoval configure -from $min -to $max

	  # set min/max values of quantity tab

        }

	
    }
    
    method get_file_name { filename } {
	set end [string length $filename]
	set start [string last "/" $filename]
	set start [expr 1 + $start]
	
	return [string range $filename $start $end]	
    }

    method build_data_info_page { w which case } {
	set page [$w add -label $which]

	iwidgets::scrolledframe $page.sf \
	    -width $notebook_width -height $notebook_height \
	    -labeltext "Data: $which" \
	    -vscrollmode dynamic \
	    -hscrollmode none \
	    -background Grey

	pack $page.sf -anchor n -fill x

	add_isosurface_section [$page.sf childsite] $which $case

    }

    method add_isosurface_section { w case} {
	global mods
	global $mods(Isosurface)-isoval
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

	iwidgets::labeledframe $w.isosurface \
	    -labelpos nw -labeltext "Isosurface"

	pack $w.isosurface -side top -anchor n -fill both -expand 1

	set isosurface [$w.isosurface childsite]
	if {$case == 0} {
	  set isosurface1 $isosurface
        } else {
	  set isosurface2 $isosurface
        }

	# isosurface sliders
	iwidgets::tabnotebook $isosurface.tnb \
            -width [expr $notebook_width-35] -height 1i \
            -tabpos n
        pack $isosurface.tnb

	
        #########
	set page [$isosurface.tnb add -label "Slider" -command "global $mods(Isosurface)-active-isoval-selection-tab; set $mods(Isosurface)-active-isoval-selection-tab 0"]

        scale $page.isoval -label "Isoval:" \
            -variable $mods(Isosurface)-isoval \
            -from [set $mods(Isosurface)-isoval-min] \
            -to [set $mods(Isosurface)-isoval-max] \
	    -length 3c \
	    -showvalue true \
            -orient horizontal \
            -digits 5 \
            -resolution 0.001 \
            -command "$this update_isosurface"
 
        pack $page.isoval -side top -anchor n

        #########
	set page [$isosurface.tnb add -label "Quantity" -command "global $mods(Isosurface)-active-isoval-selection-tab; set $mods(Isosurface)-active-isoval-selection-tab 1"]


	global $mods(Isosurface)-isoval-quantity
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max
        global $mods(Isosurface)-quantity-min
	set $mods(Isosurface)-quantity-min [set $mods(Isosurface)-isoval-min]
        global $mods(Isosurface)-quantity-max
	set $mods(Isosurface)-quantity-max [set $mods(Isosurface)-isoval-max]
	global $mods(Isosurface)-quantity-range
	set $mods(Isosurface)-quantity-range "manual"
	frame $page.q
	pack $page.q -side top -anchor n

	label $page.q.label -text "Number of evenly\nspaced isovals:"
        entry $page.q.entry -text $mods(Isosurface)-isoval-quantity 
	pack $page.q.label -side left -anchor nw
	pack $page.q.entry -side left -anchor w

	frame $page.m
	pack $page.m -side top -anchor n

	label $page.m.minl -text "Min:"
        entry $page.m.mine -text $mods(Isosurface)-quantity-min \
              -width 5
	label $page.m.maxl -text "Max:"
        entry $page.m.maxe -text $mods(Isosurface)-quantity-max \
              -width 5
	pack $page.m.minl $page.m.mine $page.m.maxl $page.m.maxe \
	     -side left -anchor nw -padx 5 -pady 5

	button $page.b -text "Extract" -width 20 \
             -command "$mods(Isosurface)-c needexecute"
	pack $page.b -side bottom -anchor s -pady 5


        #########
        set page [$isosurface.tnb add -label "List" -command "global $mods(Isosurface)-active-isoval-selection-tab; set $mods(Isosurface)-active-isoval-selection-tab 2"]


	global $mods(Isosurface)-isoval-list
	frame $page.v
	pack $page.v -side top -anchor n

	label $page.v.label -text "List of Isovals:"
        entry $page.v.entry -text $mods(Isosurface)-isoval-list \
	     -width 30
	pack $page.v.label $page.v.entry -side left -anchor nw -pady 5

	button $page.b -text "Extract" -width 20 \
             -command "$mods(Isosurface)-c needexecute"
	pack $page.b -side bottom -anchor s  -pady 5

        $isosurface.tnb view "Slider"

    }
 
    method update_progress { which state } {
	global mods
	if {$which == $mods(Isosurface)} {
           if {$state == "JustStarted 1123"} {
     	      after 1 "$standalone_progress1 reset"
     	      after 1 "$standalone_progress2 reset"
           } elseif {$state == "Executing"} {
     	      after 1 "$standalone_progress1 step"
     	      after 1 "$standalone_progress2 step"

           } elseif {$state == "NeedData"} {

           } elseif {$state == "Completed"} {
              set remaining [$standalone_progress1 cget -steps]
	      after 1 "$standalone_progress1 step $remaining"
	      after 1 "$standalone_progress2 step $remaining"


              # reconfigure min/max on slider if needed
              global $mods(Isosurface)-isoval-min
	      global $mods(Isosurface)-isoval-max
	      global $mods(Isosurface)-quantity-min
	      global $mods(Isosurface)-quantity-max
              if {$mins($current_data) != [set $mods(Isosurface)-isoval-min]} {
		$isosurface1.tnb.canvas.notebook.cs.page1.cs.isoval configure -from [set $mods(Isosurface)-isoval-min] -to [set $mods(Isosurface)-isoval-max]
		$isosurface2.tnb.canvas.notebook.cs.page1.cs.isoval configure -from [set $mods(Isosurface)-isoval-min] -to [set $mods(Isosurface)-isoval-max]

	        set $mods(Isosurface)-quantity-min [set $mods(Isosurface)-isoval-min]
	        set $mods(Isosurface)-quantity-max [set $mods(Isosurface)-isoval-max]

              }

	   } 
	}
    }

    method indicate_error { which msg_state } {
	if {$msg_state == "Error"} {
           if {$error_module == ""} {
              set error_module $which
	      # turn progress graph red
              $standalone_progress1 configure -barcolor red
              $standalone_progress1 configure -labeltext "Error"

              $standalone_progress2 configure -barcolor red
              $standalone_progress2 configure -labeltext "Error"
           }
	}
       if {$msg_state == "Reset" || $msg_state == "Remark" || \
           $msg_state == "Warning"} {
           if {$which == $error_module} {
	      set error_module ""
              $standalone_progress1 configure -barcolor green
              $standalone_progress1 configure -labeltext "$current_step..."

              $standalone_progress2 configure -barcolor green
              $standalone_progress2 configure -labeltext "$current_step..."
            }
       }
             

    }

    method block_connection { modA portA modB portB color} {
	set connection $modA
	append connection "_p$portA"
	append connection "_to_$modB"
	append connection "_p$portB"

	block_pipe $connection $modA $portA $modB $portB $color
    }

    method display_module_error {} {
	set result [$error_module displayLog]
    }

    method indicate_dynamic_compile { which mode } {

	if {$mode == "start"} {
           $standalone_progress1 configure -labeltext "Compiling..."
           # Tooltip $standalone_progress1.label "Dynamically Compiling Algorithms.\nPlease see SCIRun Developer's\nGuide for more information"
           $standalone_progress2 configure -labeltext "Compiling..."
           # Tooltip $standalone_progress2.label "Dynamically Compiling Algorithms.\nPlease see SCIRun Developer's\nGuide for more information"
        } else {
           $standalone_progress1 configure -labeltext "$current_step..."
           # Tooltip $standalone_progress1.label "Indicates current step"
           $standalone_progress2 configure -labeltext "$current_step..."
           # Tooltip $standalone_progress2.label "Indicates current step"
        }
   }


    variable eviewer
    variable win
    variable data

    # isosurface state
    variable isovals
    variable mins
    variable maxs
    variable quantity
    variable quantity_min
    variable quantity_max
    variable isoval_list
    variable iso_page

    variable data_listbox_Att
    variable data_listbox_Det
    variable isosurface1
    variable isosurface2
    variable data_frame1
    variable data_frame2
    variable notebook_width
    variable notebook_height
    variable iso_alg
    variable standalone_progress1
    variable standalone_progress2
    variable IsPAttached
    variable detachedPFr
    variable attachedPFr
    variable IsVAttached
    variable detachedVFr
    variable attachedVFr

    variable vis1
    variable vis2

    variable process_width
    variable process_height

    variable viewer_width
    variable viewer_height

    variable vis_width
    variable vis_height

    variable screen_width
    variable screen_height

    variable error_module
    variable current_step
    variable current_data

    variable data_loaded

    variable vis_tab

    variable data_label1
    variable data_label2


}

IsoApp app

app build_app






### Bind shortcuts - Must be after instantiation of IsoApp
bind all <Control-s> {
    app save_session
}

bind all <Control-o> {
    app load_session
}

bind all <Control-q> {
    app exit_app
}







