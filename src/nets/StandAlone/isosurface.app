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
set $m1-isoval-min {-5.2}
set $m1-isoval-max {4.3}
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
	toplevel .standalone
	wm title .standalone "BioIso"	 
	set win .standalone

	set notebook_width 3.5i
	set notebook_height 315

	set iso_alg 1

	set viewer_width 640
	set viewer_height 512
    
	set process_width 200
	set process_height $viewer_height

	set vis_width 300
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

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

	init_Pframe $detachedPFr.f $det_msg
	init_Pframe $attachedPFr.f $att_msg

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


    method init_Pframe { m msg } {

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
	    pack $m.p -side top -fill both -anchor n -expand 1
	    
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
	    
	    ### Progress
	    iwidgets::labeledframe $process.progress \
		-labelpos nw -labeltext "Progress" 
	    pack $process.progress -side bottom -anchor s -fill both
	    
	    set progress_section [$process.progress childsite]
	    iwidgets::feedback $progress_section.fb -labeltext "Isosurfacing..." \
		-labelpos nw \
		-steps 4 -barcolor Green \
		
	    pack $progress_section.fb -side top -padx 2 -pady 2 -anchor nw -fill both
	    
	    set iso_progress $progress_section.fb

	    ### Attach/Detach button
	    button $process.cut -text $msg -command "$this switch_P_frames"
	    pack $process.cut -side bottom -before $process.progress -anchor s -pady 5 -padx 5 

	}
	    
    }

    method init_Vframe { m msg case} {
	global mods
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization" 
	        # -background "LightSteelBlue3"
	    pack $m.vis -side top -anchor n
	    
	    set vis [$m.vis childsite]

	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height 450 -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n

	    ### Data Vis Tab
	    set page [$vis.tnb add -label "Data Vis"]
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
	    
	    $page.data insert 0 {None}
	    
	    
	    ### Data Info
	    frame $page.f -relief groove -borderwidth 2
	    pack $page.f -side top -anchor n -fill x
	    
	    iwidgets::notebook $page.f.nb -width $notebook_width \
		-height $notebook_height 
	    pack $page.f.nb -padx 4 -pady 4 -anchor n

	    if {$case == 0} {
		# detached case
		set notebook_Det $page.f.nb
	    } else {
		# attached case
		set notebook_Att $page.f.nb
	    }
	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis


	    ### Attach/Detach button
	    button $vis.cut -text $msg -command "$this switch_V_frames"
	    pack $vis.cut -side top -anchor s -pady 5 -padx 5 

	}
    }

    method create_viewer_tab { vis } {
	global mods
	set page [$vis.tnb add -label "Global Options"]
	
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
	
	checkbutton $view_opts.eframe.a.clip -text "Use Clip" \
	    -variable $mods(Viewer)-ViewWindow_0-global-clip \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.cull -text "Back Cull" \
	    -variable $mods(Viewer)-ViewWindow_0-global-cull \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.dl -text "Display List" \
	    -variable $mods(Viewer)-ViewWindow_0-global-dl \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	
	pack $view_opts.eframe.a.light $view_opts.eframe.a.fog \
	    $view_opts.eframe.a.bbox $view_opts.eframe.a.clip \
	    $view_opts.eframe.a.cull $view_opts.eframe.a.dl \
	    -side top -anchor w -padx 4 -pady 4
	
	frame $view_opts.eframe.b -relief groove -borderwidth 2
	pack $view_opts.eframe.b -side left -anchor ne -fill both
	
	label $view_opts.eframe.b.label -text "Shading:"
	pack $view_opts.eframe.b.label -side top -anchor nw
	
	radiobutton $view_opts.eframe.b.wire -text "Wire" \
	    -variable $mods(Viewer)-ViewWindow_0-global-type \
	    -value "Wire" \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw" \
	    -anchor w
	
	radiobutton $view_opts.eframe.b.flat -text "Flat" \
	    -variable $mods(Viewer)-ViewWindow_0-global-type \
	    -value "Flat" \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw" \
	    -anchor w
	
	radiobutton $view_opts.eframe.b.gouraud -text "Gouraud" \
	    -variable $mods(Viewer)-ViewWindow_0-global-type \
	    -value "Gouraud" \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw" \
	    -anchor w
	
	pack $view_opts.eframe.b.wire $view_opts.eframe.b.flat $view_opts.eframe.b.gouraud -side top -anchor nw -padx 4 -pady 4
	
	frame $view_opts.buttons -relief flat
	pack $view_opts.buttons -side top -anchor n -padx 4 -pady 4
	
	frame $view_opts.buttons.v1
	pack $view_opts.buttons.v1 -side left -anchor nw
	
	
	button $view_opts.buttons.v1.autoview -text "Autoview" \
	    -command "$mods(Viewer)-ViewWindow_0-c autoview" \
	    -width 12 -padx 3 -pady 3
	
	pack $view_opts.buttons.v1.autoview -side top -padx 3 -pady 3 \
	    -anchor n -fill x
	
	
	frame $view_opts.buttons.v1.views
	pack $view_opts.buttons.v1.views -side top -anchor nw -fill x -expand 1
	
	menubutton $view_opts.buttons.v1.views.def -text "Views" \
	    -menu $view_opts.buttons.v1.views.def.m -relief raised \
	    -padx 3 -pady 3  -width 12
	
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
	
	button $view_opts.buttons.v2.sethome -text "Set Home View" -padx 3 -pady 3 \
	    -command "$mods(Viewer)-ViewWindow_0-c sethome"
	
	button $view_opts.buttons.v2.gohome -text "Go Home" \
	    -command "$mods(Viewer)-ViewWindow_0-c gohome" \
	    -padx 3 -pady 3
	
	pack $view_opts.buttons.v2.sethome $view_opts.buttons.v2.gohome \
	    -side top -padx 2 -pady 2 -anchor ne -fill x
	
	$vis.tnb view "Data Vis"
	
    }


    method switch_P_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

	if { $IsPAttached } {	    
	    pack forget $attachedPFr
	    set new_width [expr $c_width - $process_width]
	    append geom1 $new_width x $c_height 
            wm geometry $win $geom1 
	    append geom2 $process_width x $c_height 
	    wm geometry $detachedPFr $geom2
	    wm deiconify $detachedPFr
	    set IsPAttached 0
	} else {
	    wm withdraw $detachedPFr
	    pack $attachedPFr -anchor n -side left -before $win.viewer
	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsPAttached 1
	}
	update
    }

    method switch_V_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

	if { $IsVAttached } {
	    # select data in detached data list box
	    if {[$data_listbox_Att curselection] != ""} {
		$data_listbox_Det selection set [$data_listbox_Att curselection] [$data_listbox_Att curselection]
	    }
	    pack forget $attachedVFr
	    set new_width [expr $c_width - $vis_width]
	    append geom1 $new_width x $c_height
            wm geometry $win $geom1
	    append geom2 $vis_width x $c_height
	    wm geometry $detachedVFr $geom2
	    wm deiconify $detachedVFr
	    set IsVAttached 0
	} else {
	    if {[$data_listbox_Det curselection] != ""} {
	    $data_listbox_Att selection set [$data_listbox_Det curselection] [$data_listbox_Det curselection]
	    }

	    wm withdraw $detachedVFr
	    pack $attachedVFr -anchor n -side left -after $win.viewer
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
	global mods
	global $mods(FieldReader)-filename
	# set the FieldReader filename if it hasn't been set
	if {[set $mods(FieldReader)-filename] == ""} {
	    set $mods(FieldReader)-filename $file
	    $mods(FieldReader)-c needexecute
	}
	
	# get the filename and stick into an array formated 
	# data(file) = full path
	set name [get_file_name $file]
	set data($name) $file
	
	# add the data to the listbox
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

	# build the data info notetab for this
	build_data_info_page $notebook_Att $name
	build_data_info_page $notebook_Det $name

	# bring new data info page forward
	$notebook_Att view $name
	$notebook_Det view $name
	
	# reset progress
	$iso_progress reset

    }

    method data_selected {} {
	global mods
	global $mods(FieldReader)-filename
	set current ""
	if {$IsVAttached} {
	    set current [$data_listbox_Att getcurselection]
	} else {
	    set current [$data_listbox_Det getcurselection]
	}

	if {[info exists data($current)] == 1} {

            # bring data info page forward
	    $notebook_Att view $current
	    $notebook_Det view $current
	    
	    # data selected - update FieldReader and execute
	    set $mods(FieldReader)-filename $data($current)
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

	    # Save out data information 
	    global $mods(FieldReader)-filename
	    puts $fileid "\n\#Current Data Loaded"
	    puts $fileid "set S_filename [set $mods(FieldReader)-filename]"

	    # Save out Isosurface info
 	    global $mods(Isosurface)-isoval
	    global $mods(Isosurface)-isoval-min
	    global $mods(Isosurface)-isoval-max
  	    puts $fileid "\n\#Isosurface Settings"
  	    puts $fileid "set S_isoval [set $mods(Isosurface)-isoval]"
	    puts $fileid "set S_isoval_min [set $mods(Isosurface)-isoval-min]"
	    puts $fileid "set S_isoval_max [set $mods(Isosurface)-isoval-max]"
	    puts $fileid "set S_isoval_alg $iso_alg"

	    # save_advanced $fileid $mods(Isosurface)
	    
	    # Save out global rendering properties
	    global $mods(Viewer)-ViewWindow_0-global-light
	    global $mods(Viewer)-ViewWindow_0-global-fog
	    global $mods(Viewer)-ViewWindow_0-global-debug
	    global $mods(Viewer)-ViewWindow_0-global-clip
	    global $mods(Viewer)-ViewWindow_0-global-cull
	    global $mods(Viewer)-ViewWindow_0-global-dl
	    global $mods(Viewer)-ViewWindow_0-global-type
  	    puts $fileid "\n\#Global Rendering Options"
  	    puts $fileid "set S_global_light [set $mods(Viewer)-ViewWindow_0-global-light]"
  	    puts $fileid "set S_global_fog [set $mods(Viewer)-ViewWindow_0-global-fog]"
  	    puts $fileid "set S_global_debug [set $mods(Viewer)-ViewWindow_0-global-debug]"
  	    puts $fileid "set S_global_clip [set $mods(Viewer)-ViewWindow_0-global-clip]"
  	    puts $fileid "set S_global_cull [set $mods(Viewer)-ViewWindow_0-global-cull]"
  	    puts $fileid "set S_global_dl [set $mods(Viewer)-ViewWindow_0-global-dl]"
	    puts $fileid "set S_global_type [set $mods(Viewer)-ViewWindow_0-global-type]"

	    
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

		# Load data
		load_data $S_filename

		# Set the isosurface
		global $mods(Isosurface)-isoval
		set $mods(Isosurface)-isoval $S_isoval
		set $mods(Isosurface)-isoval-min $S_isoval_min
		set $mods(Isosurface)-isoval-max $S_isoval_max
		#configure $isosurface.isoval -from $S_isoval_min \
		 	-to $S_isoval_max
		update
		set iso_alg $S_isoval_alg
		$mods(Isosurface) select-alg $iso_alg

		update_isosurface $S_isoval

		# Set the global rendering options
		global $mods(Viewer)-ViewWindow_0-global-light
		set $mods(Viewer)-ViewWindow_0-global-light $S_global_light
	}	
	
    }
        
    method exit_app {} {
	netedit quit
    }

    method show_help {} {
	puts "NEED TO IMPLEMENT SHOW HELP"
    }

    method update_isosurface_scale {} {
	# puts "Update Isosurface Scale"
	global mods

	# Update min and max for isosurface scale
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

	#UpdateIsosurface 0
	# puts [set $mods(Isosurface)-isoval-min]
	# puts [set $mods(Isosurface)-isoval-max]

	#$isosurface.isoval configure -from [set $mods(Isosurface)-isoval-min] \
	 #   -to [set $mods(Isosurface)-isoval-max]

    }

    method update_isosurface { isoval } {

	global mods

	# execute with the given value
	[$mods(Isosurface) get_this_c] needexecute

	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

    }
    
    method get_file_name { filename } {
	set end [string length $filename]
	set start [string last "/" $filename]
	set start [expr 1 + $start]
	
	return [string range $filename $start $end]	
    }

    method build_data_info_page { w which } {
	set page [$w add -label $which]

	iwidgets::scrolledframe $page.sf \
	    -width $notebook_width -height $notebook_height \
	    -labeltext "Data: $which" \
	    -vscrollmode dynamic \
	    -hscrollmode none \
	    -background Grey

	pack $page.sf -anchor n -fill x

	add_isosurface_section [$page.sf childsite] $which

    }

    method add_isosurface_section { w which } {
	global mods
	global $mods(Isosurface)-isoval
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

	iwidgets::labeledframe $w.isosurface \
	    -labelpos nw -labeltext "Isosurface"

	pack $w.isosurface -side top -anchor n -fill both -expand 1

	set isosurface [$w.isosurface childsite]

	# isosurface sliders
	scale $isosurface.isoval -label "Isoval:" \
 	    -variable $mods(Isosurface)-isoval \
 	    -from [set $mods(Isosurface)-isoval-min] \
 	    -to [set $mods(Isosurface)-isoval-max] \
 	    -length 3c \
 	    -showvalue true \
 	    -orient horizontal \
 	    -digits 5 \
 	    -resolution 0.001 \
 	    -command "$this update_isosurface" \
	
 	pack $isosurface.isoval -side top -anchor n 

	# isosurface method
	label $isosurface.mlabel -text "Method"
	pack $isosurface.mlabel -side top -anchor nw 

	frame $isosurface.method -relief flat
	pack $isosurface.method  -side top -anchor n -fill x

	radiobutton $isosurface.method.noise -text "Noise" \
	    -variable $iso_alg \
	    -value 1 \
	    -command "$mods(Isosurface) select-alg 1" \
	    -anchor w

	radiobutton $isosurface.method.mc -text "Marching Cubes" \
	    -variable $iso_alg \
	    -value 0 \
	    -command "$mods(Isosurface) select-alg 0" \
	    -anchor w

	pack $isosurface.method.noise $isosurface.method.mc -side left \
	    -anchor n -fill x

	# turn on Noise by default
	$isosurface.method.noise invoke

	
	label $isosurface.blank -text "  " -width 31
	pack $isosurface.blank -side top -anchor n
	
    }
 
    method update_progress { which state } {
	if {$which == "SCIRun_Visualization_Isosurface_0"} {
     		after 1 "$iso_progress step"
	}
        
    }

    method indicate_error {} {

    }

    variable eviewer
    variable win
    variable data
    variable data_listbox_Att
    variable data_listbox_Det
    variable isosurface
    variable notebook_Att
    variable notebook_Det
    variable notebook_width
    variable notebook_height
    variable iso_alg
    variable iso_progress
    variable IsPAttached
    variable detachedPFr
    variable attachedPFr
    variable IsVAttached
    variable detachedVFr
    variable attachedVFr

    variable process_width
    variable process_height

    variable viewer_width
    variable viewer_height

    variable vis_width
    variable vis_height

    variable screen_width
    variable screen_height

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







