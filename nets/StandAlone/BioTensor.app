# global array indexed by module name to keep track of modules
global mods


############# NET ##############
::netedit dontschedule
                                                                               
set m0 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 25 16]
set m1 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 97 98]
set m2 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 343 97]
set m3 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 97 165]
set m4 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 343 163]
set m5 [addModuleAtPosition "Teem" "Tend" "TendEpireg" 25 264]
set m6 [addModuleAtPosition "Teem" "Tend" "TendEstim" 25 643]
set m7 [addModuleAtPosition "Teem" "Tend" "TendBmat" 43 575]
set m8 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 99 347]
set m9 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 345 346]
set m10 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 99 416]
set m11 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 345 420]
set m12 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 325 248]
set m13 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 565 332]
set m14 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 327 503]
set m15 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 564 591]
set m16 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 565 395]
set m17 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 564 655]
set m18 [addModuleAtPosition "Teem" "Tend" "TendEval" 364 729]
set m19 [addModuleAtPosition "Teem" "Tend" "TendEvec" 543 729]
set m20 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 25 797]
set m21 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 185 730]
set m22 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 185 796]
set m23 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 364 796]
set m24 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 543 795]
set m25 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 26 867]
set m26 [addModuleAtPosition "SCIRun" "Render" "Viewer" 302 1021]

puts "****** Modules *******"
puts "m0 $m0"
puts "m1 $m1"
puts "m2 $m2"
puts "m3 $m3"
puts "m4 $m4"
puts "m5 $m5"
puts "m6 $m6"
puts "m7 $m7"
puts "m8 $m8"
puts "m9 $m9"
puts "m10 $m10"
puts "m11 $m11"
puts "m12 $m12"
puts "m13 $m13"
puts "m14 $m14"
puts "m15 $m15"
puts "m16 $m16"
puts "m17 $m17"
puts "m18 $m18"
puts "m19 $m19"
puts "m20 $m20"
puts "m21 $m21"
puts "m22 $m22"
puts "m23 $m23"
puts "m24 $m24"
puts "m25 $m25"
puts "m26 $m26"

	
set mods(Reader) $m0
set mods(Vol1) $m1
set mods(Vol2) $m2
set mods(Slice1) $m3
set mods(Slice2) $m4
set mods(Join1) $m12
set mods(NrrdToField1) $m13

set mods(Epireg) $m5


set mods(Viewer) $m26

                                                                               
addConnection $m0 0 $m5 0
addConnection $m0 0 $m1 0
addConnection $m0 0 $m2 0
addConnection $m1 0 $m3 0
addConnection $m2 0 $m4 0
addConnection $m9 0 $m11 0
addConnection $m8 0 $m10 0
addConnection $m5 0 $m8 0
addConnection $m5 0 $m9 0
addConnection $m5 0 $m6 0
addConnection $m12 0 $m13 0
addConnection $m4 0 $m12 0
addConnection $m3 0 $m12 1
addConnection $m4 0 $m12 2
addConnection $m11 0 $m14 0
addConnection $m10 0 $m14 1
addConnection $m11 0 $m14 2
addConnection $m14 0 $m15 0
addConnection $m13 0 $m16 0
addConnection $m15 0 $m17 0
addConnection $m6 0 $m18 0
addConnection $m6 0 $m19 0
addConnection $m6 0 $m20 0
addConnection $m6 0 $m21 0
addConnection $m21 0 $m22 0
addConnection $m18 0 $m23 0
addConnection $m19 0 $m24 0
addConnection $m7 0 $m6 1

#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

class BioTensorApp {

    method modname {} {
	return "BioTensorApp"
    }

    constructor {} {
	toplevel .standalone
	wm title .standalone "BioTensor"	 
	set win .standalone

	set notebook_width 300
	set notebook_height 315

	set viewer_width 640
	set viewer_height 512
    
	set process_width 300
	set process_height $viewer_height

	set vis_width [expr $notebook_width + 40]
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

        # Dummy variables
        set number_of_images 0
        set file_prefix "/scratch/darbyb/data/img"
        set threshold 270
        set flood_fill 1

        set dt1 ""
        set dt2 ""

        set error_module ""

        set current_step "Data Acquisition"

        set proc_color "dark red"

        initialize_blocks

    }

    destructor {
	destroy $this
    }

    method initialize_blocks {} { 
	global mods

        # Blocking Data Section
	block_connection $mods(Reader) 0 $mods(Vol1) 0 "purple"
	block_connection $mods(Vol1) 0 $mods(Slice1) 0 "purple"

	block_connection $mods(Reader) 0 $mods(Vol2) 0 "purple"
	block_connection $mods(Vol2) 0 $mods(Slice2) 0 "purple"

	block_connection $mods(Slice2) 0 $mods(Join1) 0 "purple"
	block_connection $mods(Slice1) 0 $mods(Join1) 1 "purple"
	block_connection $mods(Slice2) 0 $mods(Join1) 2 "purple"

	block_connection $mods(Reader) 0 $mods(Epireg) 0 "purple"

        block_connection $mods(Join1) 0 $mods(NrrdToField1) 0 "purple"

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

	### set frame data members
	set detachedPFr $win.detachedP
	set attachedPFr $win.attachedP

	init_Pframe $detachedPFr.f $det_msg 0
	init_Pframe $attachedPFr.f $att_msg 1

	### create detached width and heigh
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

	### set frame data members
	set detachedVFr $win.detachedV
	set attachedVFr $win.attachedV
	
	init_Vframe $detachedVFr.f $det_msg 0
	init_Vframe $attachedVFr.f $att_msg 1



	### pack 3 frames
	pack $attachedPFr $win.viewer $attachedVFr -side left \
	     -anchor n -fill both -expand 1

	set total_width [expr [expr $process_width + $viewer_width] + $vis_width]
	set total_height $vis_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $viewer_height + $pos_x + $pos_y
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
	    pack $m.p -side left -fill both -anchor n -expand 1
	    
	    set process [$m.p childsite]
	    
	    ### Data Acquisition
	    iwidgets::labeledframe $process.data \
		-labelpos nw -labeltext "Data Acquisition" \
                -foreground $proc_color -borderwidth 5 -relief raised
	    pack $process.data -side top -fill both -anchor n

            if {$case == 0} {
               set steps1(Data) $process.data
            } else {
               set steps2(Data) $process.data
            }
	    
	    set step [$process.data childsite]

	    
	    frame $step.n 
            pack $step.n -side top -anchor nw
	
	    label $step.n.label -text "Number of Images:"
            entry $step.n.entry -textvariable $number_of_images -width 15
            pack $step.n.label $step.n.entry -side left -anchor n \
                  -padx 5 -pady 5 -fill x -expand 1

            frame $step.p
            pack $step.p -side top -anchor nw
     
            label $step.p.label -text "File Prefix:"
            entry $step.p.entry -textvariable $file_prefix -width 22
            pack $step.p.label $step.p.entry -side left -anchor n \
	         -padx 5 -pady 5 -fill x -expand 1

	    frame $step.b1
	    pack $step.b1 -side top -anchor n -fill x -expand 1
            
	    button $step.b1.ex -text "Next" -command "$this execute_DataAcquisition" -width 15
            pack $step.b1.ex  -side top -anchor n \
		-padx 5 -pady 5 
	
	    ### Registration
	    iwidgets::labeledframe $process.reg \
               -labelpos nw -labeltext "Registration" 
            pack $process.reg -side top -fill both -anchor n

            if {$case == 0} {
               set steps1(Registration) $process.reg
            } else {
               set steps2(Registration) $process.reg
            }

	    set step [$process.reg childsite]

	    if {$case == 0} {
                 set registration1 $step
            } else {
                 set registration2 $step
            }

	    frame $step.t
            pack $step.t -side top -anchor nw -fill x -expand 1
            
            label $step.t.label -text "Threshold:" -state disabled
	    entry $step.t.entry -textvariable $threshold -state disabled
	    pack $step.t.label $step.t.entry -side left -anchor n \
	         -padx 5 -pady 5 -fill x -expand 1
	
	    checkbutton $step.flood -text "Use Flood Fill Method" \
                 -variable $flood_fill -state disabled
            pack $step.flood -side top -anchor nw -padx 5 -pady 5 

	    frame $step.b1
	    pack $step.b1 -side top -anchor n -fill x -expand 1
            
	    button $step.b1.ex -text "Next" -state disabled -width 15\
	         -command "$this execute_registration"
            pack $step.b1.ex -side top -anchor n \
                 -padx 5 -pady 5
              
	    ### Build DT
            iwidgets::labeledframe $process.dt \
                 -labelpos nw -labeltext "Build Diffusion Tensors"
            pack $process.dt -side top -fill both -anchor n

            if {$case == 0} {
               set steps1(DT) $process.dt
            } else {
               set steps2(DT) $process.dt
            }


            set step [$process.dt childsite]
	    if {$case == 0} {
	         set dt1 $step
            } else {
	         set dt2 $step
            }


	    frame $step.b1
	    pack $step.b1 -side top -anchor nw -fill x -expand 1

            button $step.b1.ex -text "Next" -state disabled -width 15 \
                 -command "$this execute_dt"
            pack $step.b1.ex -side top -anchor n \
                 -padx 5 -pady 5 


	    ### Progress
	    iwidgets::labeledframe $process.progress \
		-labelpos nw -labeltext "Progress" 
	    pack $process.progress -side bottom -anchor s -fill both
	    
	    set progress_section [$process.progress childsite]
	    iwidgets::feedback $progress_section.fb -labeltext "$current_step..." \
		-labelpos nw \
		-steps 10 -barcolor Green \
		
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

	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height 490 -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

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
	    pack $page.f.nb -padx 4 -pady 4 -anchor n -fill both -expand 1

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
            frame $m.d 
	    pack $m.d -side left -anchor e
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
	       -fill both -expand 1
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
	       -fill both -expand 1
	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
	update
    }

    method add_data { file } {
	
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
	$standalone_progress reset

    }

    method data_selected {} {
	global mods

	set current_data ""
	if {$IsVAttached} {
	    set current_data [$data_listbox_Att getcurselection]
	} else {
	    set current_data [$data_listbox_Det getcurselection]
	}
	if {[info exists data($current_data)] == 1} {

            # bring data info page forward
	    $notebook_Att view $current_data
	    $notebook_Det view $current_data
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
	    puts $fileid "Save\n"
	    
	    close $fileid
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

	}	
	
    }
        
    method exit_app {} {
	netedit quit
    }

    method show_help {} {
	puts "NEED TO IMPLEMENT SHOW HELP"
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
	    -hscrollmode dynamic \
	    -background Grey

	pack $page.sf -anchor n -fill x

    }

    method update_progress { which state } {

	global mods
        if {$current_step == "Data Acquisition"} {
	if {$which == $mods(Reader)} {
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

    method execute_DataAcquisition {} {
       global mods 

       # execute modules
       $mods(Join1)-c needexecute       

       if {$current_step == "Data Acquisition"} {
          # un-highlight current processing step
          $steps1(Data) configure -borderwidth 2 -foreground "black" -relief groove
          $steps2(Data) configure -borderwidth 2 -foreground "black" -relief groove

          # change button label from next to execute
          [$steps1(Data) childsite].b1.ex configure -text "Execute"
          [$steps2(Data) childsite].b1.ex configure -text "Execute"

          # activate next processing step
          activate_registration

          # highlight new processing step
          $steps1(Registration) configure -borderwidth 5 -foreground $proc_color -relief raised
          $steps2(Registration) configure -borderwidth 5 -foreground $proc_color -relief raised
       }

    }
    method activate_registration { } {
        puts "change data acquisition label color and borderwidth back"
        puts "Configure registration label color and borderwidth"
     
	foreach w [winfo children $registration1] {
	    activate_widget $w
        }

	foreach w [winfo children $registration2] {
	    activate_widget $w
        }

        set current_step "Registration"
        $standalone_progress1 configure -labeltext "$current_step..."
        $standalone_progress2 configure -labeltext "$current_step..."
 
        puts "unblock registration connections"
    }


    method execute_registration {} {
	global mods
 
        # execute modules

        if {$current_step == "Registration"} {
           $steps1(Registration) configure -borderwidth 2 -foreground "black" -relief groove
           $steps2(Registration) configure -borderwidth 2 -foreground "black" -relief groove

           # change button label from next to execute
           [$steps1(Registration) childsite].b1.ex configure -text "Execute"
           [$steps2(Registration) childsite].b1.ex configure -text "Execute"

           activate_dt 

           $steps1(DT) configure -borderwidth 5 -foreground $proc_color -relief raised
           $steps2(DT) configure -borderwidth 5 -foreground $proc_color -relief raised
        }

    }


    method activate_dt { } {
        puts "change data registration color and borderwidth back"
	puts "hightlight vis section???"

	foreach w [winfo children $dt1] {
	    activate_widget $w
        }

	foreach w [winfo children $dt2] {
	    activate_widget $w
        }

        set current_step "Building Diffusion Tensors"
        $standalone_progress1 configure -labeltext "$current_step..."
        $standalone_progress2 configure -labeltext "$current_step..."

        puts "unblock dt connections"
    }


    method execute_dt {} {

       if {$current_step == "Building Diffusion Tensors"} {
          $steps1(DT) configure -borderwidth 2 -foreground "black" -relief groove
          $steps2(DT) configure -borderwidth 2 -foreground "black" -relief groove

          # change button label from next to execute
          [$steps1(DT) childsite].b1.ex configure -text "Execute"
          [$steps2(DT) childsite].b1.ex configure -text "Execute"
       }
    }

    method activate_widget {w} {
    	set has_state_option 0
    	foreach opt [$w configure ] {
	    set temp [lsearch -exact $opt "state"]
	    if {$temp > -1} {
	       set has_state_option 1
	    }
        }
        if {$has_state_option} {
	    $w configure -state normal
        }

        foreach widg [winfo children $w] {
	     activate_widget $widg
        }
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

    # pointers to steps
    variable registration1
    variable registration2
    variable dt1
    variable dt2

    variable data_listbox_Att
    variable data_listbox_Det

    variable notebook_Att
    variable notebook_Det
    variable notebook_width
    variable notebook_height

    variable standalone_progress1
    variable standalone_progress2

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

    variable error_module

    variable current_step

    variable proc_color

    variable steps1
    variable steps2

    # Dummy Variables
    variable number_of_images
    variable file_prefix
    variable threshold
    variable flood_fill

}

BioTensorApp app

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


