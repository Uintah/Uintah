class PowerAppBase {
    ############################
    ### constructor
    ############################
    # The PowerAppBase constructor will be called before the child's constructor.
    # It will initialize variables regarding size, state, the progress indicator, tooltips, etc.
    # The constructor also configures the scheme (color, fonts, etc)
    constructor {} {
	configure_scheme 

	# Standalone and viewer window
	set win .standalone
	set eviewer ""

	# Window sizes
	set process_width 0
	set process_height 0
	
	set viewer_width 0
	set viewer_height 0
	
	set vis_width 0
	set vis_height 0

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

	# Attached/detached frames
	set IsPAttached 1
	set detachedPFr ""
	set attachedPFr ""

	set IsVAttached 1
	set detachedVFr "" 
	set attachedVFr ""


	# State
	set initialized 0
	set loading 0

	# Indicator variables
        set indicator1 ""
        set indicator2 ""
        set indicatorL1 ""
        set indicatorL2 ""
        set indicatorID 0
        set indicate 0
        set cycle 0
	set executing_modules 0
        set i_width 220
        set i_height 20
        set stripes 8
        set i_move [expr [expr $i_width/double($stripes)]/2.0]
        set i_back [expr $i_move*-3]
        set error_module ""

	# Colormaps
        set colormap_width 80
        set colormap_height 15
        set colormap_res 64

	global tips
 	# Menu
	set tips(FileMenu) [subst {\
        Save Session...  \tSave a BioTensor session\n\
                         \t\tto load at a later time\n\
        Load Session...  \tLoad a BioTensor session\n\
        Quit             \tQuit BioTensor} ]

	set tips(HelpMenu) [subst {\
        Show Tooltips   \tTurn tooltips on or off\n\
        Help Contents   \tHelp for BioTensor\n\
        About BioTensor \tInformation about\n\
		        \t\tBioTensor } ]
 	# Indicator
 	set tips(IndicatorBar) \
 	    "Indicates the status of\napplication. Click when\nred to view error message."
 	set tips(IndicatorLabel) \
 	    "Indicates the current\nstep in progress."

	# Viewer Options Tab
	set tips(ViewerLighting) \
	    "Toggle whether or not the\nViewer applies lighting to\nthe display. Objects\nwithout lighting have a\nconstant color."
	set tips(ViewerFog) \
	    "Toggle to draw objects\nwith variable intensity\nbased on their distance\nfrom the user. Also\nknown as depth cueing.\nClose objects appear\nbrighter."
	set tips(ViewerBBox) \
	    "Toggle whether the Viewer\ndraws the selected objects\nin full detail or as a simple\nbounding box."
	set tips(ViewerCull) \
	    "Display only the forward\nfacing facets."
	set tips(ViewerSetHome) \
	    "Captures the current view\nso the user can return to\nit later by clicking Go Home."
	set tips(ViewerGoHome) \
	    "Restores the\ncurrent\nhome view."
	set tips(ViewerViews) \
	    "Lists a number of\nstandard viewing\nangles and orientations."
	set tips(ViewerAutoview) \
	    "Restores the viewer to\nthe default condition."

 	# Attach/Detach Mouseovers
 	set tips(ProcAttachHashes) "Click hash marks to\nattach to Viewer."
 	set tips(ProcDetachHashes) "Click hash marks to detach\nfrom the Viewer."
 	set tips(VisAttachHashes) "Click hash marks to\nattach to Viewer."
 	set tips(VisDetachHashes) "Click hash marks to detach\nfrom the Viewer."	
	
    }

    ############################
    ### destructor
    #############################
    destructor {

    }

    ############################
    ### appname
    ############################
    # Returns the name of the app
    method appname {} {
	return "PowerAppBase"
    }	


    #############################
    ### configure_scheme
    #############################
    # Configure the color scheme and look and feel to be the same.
    # This includes the colors for the next and execute buttons.
    method configure_scheme {} {
	set basecolor grey
	
	. configure -background $basecolor
	
	option add *Frame*background black
	
	option add *Button*padX 1
	option add *Button*padY 1
	
	option add *background $basecolor
	option add *activeBackground $basecolor
	option add *sliderForeground $basecolor
	option add *troughColor $basecolor
	option add *activeForeground white
	
	option add *Scrollbar*activeBackground $basecolor
	option add *Scrollbar*foreground $basecolor
	option add *Scrollbar*width .35c
	option add *Scale*width .35c
	
	option add *selectBackground "white"
	option add *selector red
	option add *font "-Adobe-Helvetica-normal-R-Normal-*-10-120-75-*"
	option add *Labeledframe.labelFont "-Adobe-Helvetica-bold-R-Normal-*-10-120-75-*"
	option add *Entryfield.labelFont "-Adobe-Helvetica-bold-R-Normal-*-10-120-75-*"
	option add *Optionmenu.labelFont "-Adobe-Helvetica-bold-R-Normal-*-10-120-75-*"
	option add *highlightThickness 0

	set next_color "#cdc858"
	set execute_color "#5377b5"
    }


    ###############################
    ### build_menu
    ###############################
    # Build the standard menu in the specified frame.
    # The menu contains the File->Load Session, File->Save Session, etc..
    method build_menu { m } {
	global tips
	frame $m.main_menu -relief raised -borderwidth 3
	pack $m.main_menu -fill x -anchor nw
	
	menubutton $m.main_menu.file -text "File" -underline 0 \
	    -menu $m.main_menu.file.menu
	
	Tooltip $m.main_menu.file $tips(FileMenu)
	
	menu $m.main_menu.file.menu -tearoff false
	
	$m.main_menu.file.menu add command -label "Load Session...  Ctr+O" \
	    -underline 1 -command "$this load_session" -state active
	
	$m.main_menu.file.menu add command -label "Save Session... Ctr+S" \
	    -underline 0 -command "$this save_session" -state active
	
	# $m.main_menu.file.menu add command -label "Save Image..." \
	    # -underline 0 -command "$mods(Viewer)-ViewWindow_0 makeSaveImagePopup" -state active
	
	$m.main_menu.file.menu add command -label "Quit        Ctr+Q" \
	    -underline 0 -command "$this exit_app" -state active
	
	pack $m.main_menu.file -side left
	
	
	global tooltipsOn
	menubutton $m.main_menu.help -text "Help" -underline 0 \
	    -menu $m.main_menu.help.menu
	
	Tooltip $m.main_menu.help $tips(HelpMenu)
	
	menu $m.main_menu.help.menu -tearoff false
	
	$m.main_menu.help.menu add check -label "Show Tooltips" \
	    -variable tooltipsOn \
	    -underline 0 -state active
	
	$m.main_menu.help.menu add command -label "Help Contents" \
	    -underline 0 -command "$this show_help" -state active
	
	$m.main_menu.help.menu add command -label "About BioTensor" \
	    -underline 0 -command "$this show_about" -state active
	
	pack $m.main_menu.help -side left
	
	tk_menuBar $m.main_menu $m.main_menu.file $m.main_menu.help
	    
    }


    ##############################
    ### load_session
    ##############################
    # To be filled in by child class. This method should
    # control loading a sesson for a specific app.
    method load_session {} {
	puts "Define load_session for [appname] app"
    }


    #########################
    ### reset_app
    #########################
    # Method that should be called when loading a session.  
    # This just enables any disabled modules so that the
    # loaded session can disable a fresh set of modules.
    method reset_app {} {
	global mods
	# enable all modules
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    disableModule $mods($m) 0
	}
	array donesearch mods $searchID
    }
    

    ##############################
    ### save_session
    ##############################
    # To be filled in by child class. It should save out a session
    # for the specific app.
    method save_session {} {
	puts "Define save_session for [appname] app"
    }

    ##########################
    ### save_module_variables
    ##########################
    # This method saves out the variables of all of the modules to the
    # specified file. It currently only saves out the variables for the
    # modules that the application has included in the global mods array.
    method save_module_variables { fileid } {
	# make globals accessible
	foreach g [info globals] {
	    global $g
	}
	
	puts $fileid "# Save out module variables\n"
	
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    foreach v [info vars $mods($m)*] {
		set var [get_module_variable_name $v]
		if {$var != "msgStream" && ![array exists $v]} {
		    puts $fileid "set \$mods($m)-$var \{[set $mods($m)-$var]\}"
		}
	    }
	}
	array donesearch mods $searchID
    }

    ##########################
    ### get_module_variable_name
    ##########################
    # This method strips away the module name information and gets
    # just the variable name (i.e. port-index from 
    # SCIRun_FieldsOther_ChooseField_0-port-index
    method get_module_variable_name { var } {
	# take out the module part of the variable name
	set end [string length $var]
	set start [string first "-" $var]
	set start [expr 1 + $start]
	
	return [string range $var $start $end]
    }

    #########################
    ### save_disabled_modules
    #########################
    # Save out the call to disable all modules that are currently disabled
    method save_disabled_modules { fileid } {
	global mods Disabled
	
	puts $fileid "\n# Disabled Modules\n"
	
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    if {[info exists Disabled($mods($m))] && $Disabled($mods($m))} {
		puts $fileid "disableModule \$mods($m) 1"
	    }
	}
	array donesearch mods $searchID
    }
    

    #########################
    ### save_class_variables
    #########################
    # Save out all of the class variables 
    method save_class_variables { fileid } {
	puts $fileid "\n# Class Variables\n"
	
	foreach v [info variable] {
	    set var [get_class_variable_name $v]
	    if {$var != "this" } {
		puts $fileid "set $var \{[set $var]\}"
	    }
	}
	puts $fileid "set loading 1"
    }

    
    #######################
    ### get_class_variable_name
    #######################
    # Remove the :: fromt the variable
    method get_class_variable_name { var } {
	set end [string length $var]
	set start [string last "::" $var]
	set start [expr 2 + $start]
	
	return [string range $var $start $end]
    }


    ##############################
    ### show_help
    ##############################
    # To be filled in by child class
    method show_help    {} {
	puts "Define show_help for [appname] app"
    }

    ##############################
    ### show_about
    ##############################
    # To be filled in by child class
    method show_about {} {
	puts "Define show_about for [appname] app"
    }


    #############################
    ### exit_app
    #############################
    # Exit by a NiceQuit
    method exit_app {} {
	NiceQuit
    }


    #############################
    ### create_viewer_tab
    #############################
    # Build the Viewer tab.  This is actually labeled the "Viewer Options"
    method create_viewer_tab { vis } {
	global tips
	global mods
	set page [$vis.tnb add -label "Viewer Options" -command "$this change_vis_frame \"Viewer Options\""]
	
	iwidgets::labeledframe $page.viewer_opts \
	    -labelpos nw -labeltext "Global Render Options"
	
	pack $page.viewer_opts -side top -anchor n -fill both -expand 1
	
	set view_opts [$page.viewer_opts childsite]
	
	frame $view_opts.eframe -relief groove -borderwidth 2
	pack $view_opts.eframe -side top -anchor n -padx 4 -pady 4
	
	checkbutton $view_opts.eframe.light -text "Lighting" \
	    -variable $mods(Viewer)-ViewWindow_0-global-light \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	Tooltip $view_opts.eframe.light $tips(ViewerLighting)
	
	checkbutton $view_opts.eframe.fog -text "Fog" \
	    -variable $mods(Viewer)-ViewWindow_0-global-fog \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	Tooltip $view_opts.eframe.fog $tips(ViewerFog)
	
	checkbutton $view_opts.eframe.bbox -text "BBox" \
	    -variable $mods(Viewer)-ViewWindow_0-global-debug \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	Tooltip $view_opts.eframe.bbox $tips(ViewerBBox)

	checkbutton $view_opts.eframe.cull -text "Back Cull" \
	    -variable $mods(Viewer)-ViewWindow_0-global-cull \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	Tooltip $view_opts.eframe.cull $tips(ViewerCull)
	
	pack $view_opts.eframe.light $view_opts.eframe.fog \
	    $view_opts.eframe.bbox  $view_opts.eframe.cull\
	    -side left -anchor n -padx 4 -pady 4
	
	
	frame $view_opts.buttons -relief flat
	pack $view_opts.buttons -side top -anchor n -padx 4 -pady 4
	
	frame $view_opts.buttons.v1
	pack $view_opts.buttons.v1 -side left -anchor nw
	
	
	button $view_opts.buttons.v1.autoview -text "Autoview (Ctrl-v)" \
	    -command "$mods(Viewer)-ViewWindow_0-c autoview" \
	    -width 15 -padx 3 -pady 3
	Tooltip $view_opts.buttons.v1.autoview $tips(ViewerAutoview)
	
	pack $view_opts.buttons.v1.autoview -side top -padx 3 -pady 3 \
	    -anchor n -fill x
	
	
	frame $view_opts.buttons.v1.views
	pack $view_opts.buttons.v1.views -side top -anchor nw -fill x -expand 1
	
	menubutton $view_opts.buttons.v1.views.def -text "Views" \
	    -menu $view_opts.buttons.v1.views.def.m -relief raised \
	    -padx 3 -pady 3  -width 15
	Tooltip $view_opts.buttons.v1.views.def $tips(ViewerViews)
	
	menu $view_opts.buttons.v1.views.def.m -tearoff 0

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
	
	menu $view_opts.buttons.v1.views.def.m.posx -tearoff 0
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
	
	menu $view_opts.buttons.v1.views.def.m.posy -tearoff 0
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
	
	menu $view_opts.buttons.v1.views.def.m.posz -tearoff 0
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
	
	menu $view_opts.buttons.v1.views.def.m.negx -tearoff 0
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
	
	menu $view_opts.buttons.v1.views.def.m.negy -tearoff 0
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
	
	menu $view_opts.buttons.v1.views.def.m.negz -tearoff 0
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
	    -command "$mods(Viewer)-ViewWindow_0-c sethome" -width 15

	Tooltip $view_opts.buttons.v2.sethome $tips(ViewerSetHome)
	
	button $view_opts.buttons.v2.gohome -text "Go Home" \
	    -command "$mods(Viewer)-ViewWindow_0-c gohome" \
	    -padx 3 -pady 3 -width 15
	Tooltip $view_opts.buttons.v2.gohome $tips(ViewerGoHome)
	
	pack $view_opts.buttons.v2.sethome $view_opts.buttons.v2.gohome \
	    -side top -padx 2 -pady 2 -anchor ne -fill x
	
	$vis.tnb view "Vis Options"
    }

    ########################
    ### display_module_error
    ########################
    # If a module has received an error, and the
    # user clicks on the red progress bar, this
    # displays the module's log.
    method display_module_error {} {
        if {$error_module != ""} {
	    set result [$error_module displayLog]
        }
    }


    #######################
    ### indicate_dynamic_complile
    #######################
    # To be filled in by child class. It should set the indicate value
    # using the change_indicate_val call and change the indicator
    # labels with the change_indicator_labels call.
    method indicate_dynamic_compile { which mode } {
	puts "Define indicate_dynamic_compile for [appname] app"
    }


    ##########################
    ### update_progress
    ##########################
    # This is called when any module calls update_state.
    # We only care about "JustStarted" and "Completed" calls.
    # Good modules to watch are ones that define a processing step
    # and the ShowFields for visualization. Depending on the state,
    # call a change_indicate_val with a 1 (JustStarting) or 2 (Completed)
    method update_progress { which state } {
	puts "Define update_progress for [appname] app"
    }


    ##########################
    ### indicate_error
    ##########################
    # This method should change the indicator and labels to
    # the error state.  This should be done using the change_indicate_val
    # and change_indicator_labels methods.
    method indicate_error { which msg_state } {
	puts "Define indicate_error for [appnaem] app"
    }



    ###########################
    ### activate_widget
    ###########################
    # This method activates a widget and changes the foreground color to be black.
    # If it is a Next or Execute button, it changes the foreground color to be
    # the appropriate color
    method activate_widget {w} {
    	set has_state_option 0
    	set has_foreground_option 0
        set has_text_option 0
    	foreach opt [$w configure ] {
	    set temp1 [lsearch -exact $opt "state"]
	    set temp2 [lsearch -exact $opt "foreground"]
	    set temp3 [lsearch -exact $opt "text"]

	    if {$temp1 > -1} {
	       set has_state_option 1
	    }
            if {$temp2 > -1} {
               set has_foreground_option 1
            }
            if {$temp3 > -1} {
               set has_text_option 1
            }
        }

        if {$has_state_option} {
	    $w configure -state normal
        }

        if {$has_foreground_option} {
            $w configure -foreground black
        }
      
        if {$has_text_option} {
           # if it is a next button configure the background 
           set t [$w configure -text]
           if {[lindex $t 4]== "Next"} {
             $w configure -background $next_color
             $w configure -activebackground $next_color
           } elseif {[lindex $t 4] == "Execute"} {
             $w configure -background $execute_color
             $w configure -activebackground $execute_color
           }
        }

        foreach widg [winfo children $w] {
	     activate_widget $widg
        }
    }

    
    ##############################
    ### disable_widget
    ##############################
    # This method disables a widget and sets the foreground to be grey64
    method disable_widget {w} {
    	set has_state_option 0
    	set has_foreground_option 0
    	foreach opt [$w configure ] {
	    set temp1 [lsearch -exact $opt "state"]
	    set temp2 [lsearch -exact $opt "foreground"]
	    if {$temp1 > -1} {
	       set has_state_option 1
	    }
            if {$temp2 > -1} {
               set has_foreground_option 1
            }
        }

        if {$has_state_option} {
	    $w configure -state disabled
        }
        if {$has_foreground_option} {
            $w configure -foreground grey64
        }


        foreach widg [winfo children $w] {
	     disable_widget $widg
        }
    }


    ############################
    ### construct_indicator
    ############################
    # This intializes the canvases for the different
    # indicate states (reset, error, executing, complete)
    method construct_indicator { canvas } {
	global tips
	
	# make image swirl
	set dx [expr $i_width/double($stripes)]
	set x 0
	set longer [expr $stripes+10]
	for {set i 0} {$i <= $longer} {incr i 1} {
	    if {[expr $i % 2] != 0} {
		set r 83
		set g 119
		set b 181
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		set prevx [expr $oldx - $dx]
		$canvas create polygon \
		    $oldx 0 $x 0 $oldx $i_height $prevx $i_height \
		    -fill $c -outline $c -tags swirl
	    } else {
		set r 237
		set g 240
		set b 242
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		set prevx [expr $oldx - $dx]
		$canvas create polygon \
		    $oldx 0 $x 0 $oldx $i_height $prevx $i_height \
		    -fill $c -outline $c -tags swirl
	    }
	}
	
	set i_font "-Adobe-Helvetica-Bold-R-Normal-*-14-120-75-*"
	
	# make completed
	set s [expr $i_width/2]
	set dx [expr $i_width/double($s)]
	set x 0
	for {set i 0} {$i <= $s} {incr i 1} {
	    if {[expr $i % 2] != 0} {
		set r 0
		set g 139
		set b 69
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		$canvas create rectangle \
		    $oldx 0 $x $i_height \
		    -fill $c -outline $c -tags comp1
	    } else {
		set r 49
		set g 160
		set b 101
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		$canvas create rectangle \
		    $oldx 0 $x $i_height  \
		    -fill $c -outline $c -tags comp1
	    }
	}
	
	$canvas create text [expr $i_width/2] [expr $i_height/2] -text "C O M P L E T E" \
	    -font $i_font -fill "black" -tags comp2
	
	# make error
	set s [expr $i_width/2]
	set dx [expr $i_width/double($s)]
	set x 0
	for {set i 0} {$i <= $s} {incr i 1} {
	    if {[expr $i % 2] == 0} {
		set r 191
		set g 59
		set b 59
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		$canvas create rectangle \
		    $oldx 0 $x $i_height \
		    -fill $c -outline $c -tags error1
	    } else {
		set r 206
		set g 78
		set b 78
		set c [format "#%02x%02x%02x" $r $g $b]
		set oldx $x
		set x [expr ($i+1)*$dx]
		$canvas create rectangle \
		    $oldx 0 $x $i_height  \
		    -fill $c -outline $c -tags error1
	    }
	}

	$canvas create text [expr $i_width/2] [expr $i_height/2] -text "E R R O R" \
	    -font $i_font -fill "black" -tags error2
	
	# make reset
	set r 237
	set g 240
	set b 242
	set c [format "#%02x%02x%02x" $r $g $b]
	$canvas create rectangle \
	    0 0 $i_width $i_height -fill $c -outline $c -tags res
	
	bind $canvas <ButtonPress> {app display_module_error}
	
	Tooltip $canvas $tips(IndicatorBar)
    }


    
    #######################
    ### change_indicator
    #######################
    # Change the indicator bar.  If indicate equals 0,
    # reset, 1 equals start swirl, 2 equals complete,
    # 3 equals error state. Each time this executes with
    # indicate=1, the swirl canvas shifts.  After 3 cyles
    # it starts over. Only when indicate is a 1 does this
    # function call itself again.
    method change_indicator {} {
	if {[winfo exists $indicator2] == 1} {
	    
	    if {$indicatorID != 0} {
		after cancel $indicatorID
		set indicatorID 0
	    }
	    
	    if {$indicate == 0} {
		# reset and do nothing
		$indicator1 raise res all
		$indicator2 raise res all
		after cancel $indicatorID
	    } elseif {$indicate == 1} {
		# indicate something is happening
		if {$cycle == 0} { 
		    $indicator1 raise swirl all
		    $indicator2 raise swirl all
		    $indicator1 move swirl $i_back 0
		    $indicator2 move swirl $i_back 0		  
		    set cycle 1
		} elseif {$cycle == 1} {
		    $indicator1 move swirl $i_move 0
		    $indicator2 move swirl $i_move 0
		    set cycle 2
		} elseif {$cycle == 2} {
		    $indicator1 move swirl $i_move 0
		    $indicator2 move swirl $i_move 0
		    set cycle 3
		} else {
		    $indicator1 move swirl $i_move 0
		    $indicator2 move swirl $i_move 0
		    set cycle 0
		} 
		set indicatorID [after 200 "$this change_indicator"]
	    } elseif {$indicate == 2} {
		# indicate complete
		$indicator1 raise comp1 all
		$indicator2 raise comp1 all
		
		$indicator1 raise comp2 all
		$indicator2 raise comp2 all
	    } else {
		$indicator1 raise error1 all
		$indicator2 raise error1 all
		
		$indicator1 raise error2 all
		$indicator2 raise error2 all
		after cancel $indicatorID
	    }
	}
    }



    ########################
    ### change_indicate_val
    ########################
    # This will change the value of indicate if it is not in error mode.
    # This should probably be implmented by the child class so that the labels
    # can be set properly
    method change_indicate_val { v } {
	# only change an error state if it has been cleared (error_module empty)
	# it will be changed by the indicate_error method when fixed
	if {$indicate != 3 || $error_module == ""} {
	    if {$v == 3} {
		# Error
		set cycle 0
		set indicate 3
		change_indicator
	    } elseif {$v == 0} {
		# Reset
		set cycle 0
		set indicate 0
		change_indicator
	    } elseif {$v == 1} {
		# Start
		set executing_modules [expr $executing_modules + 1]
		set indicate 1
		change_indicator
	    } elseif {$v == 2} {
		# Complete
		set executing_modules [expr $executing_modules - 1]
		if {$executing_modules == 0} {
		    # only change indicator if progress isn't running
		    set indicate 2
		    change_indicator
		}
	    }
	}
    }
    

    ###########################
    ### change_indicator_labels
    ###########################
    # Implement in child class
    method change_indicator_labels { msg } {
       	puts "Define change_indicator_labels for [appname] app"
    }


    ###########################
    ### block_connection
    ###########################
    # Blocks a module connection
    method block_connection { modA portA modB portB } {
	disableConnection "$modA $portA $modB $portB"
    }



    ###########################
    ### unblock_connection
    ###########################
    # Unblocks a module connection
    method unblock_connection { modA portA modB portB } {
	disableConnection "$modA $portA $modB $portB"
    }



    
    
    #############################
    ### draw_colormap
    #############################
    # This draws a small colormap specified by
    # which on the canvas
    method draw_colormap { which canvas } {
	set color ""
	if {$which == "Gray"} {
	    set color { "Gray" { { 0 0 0 } { 255 255 255 } } }
	} elseif {$which == "Rainbow"} {
	    set color { "Rainbow" {	
		{ 255 0 0}  { 255 102 0}
		{ 255 204 0}  { 255 234 0}
		{ 204 255 0}  { 102 255 0}
		{ 0 255 0}    { 0 255 102}
		{ 0 255 204}  { 0 204 255}
		{ 0 102 255}  { 0 0 255}}}
	} elseif {$which == "Inverse Rainbow"} {
	    set color { "Inverse Rainbow" {	
		{ 0 0 255} { 0 102 255}
		{ 0 204 255} { 0 255 204}
		{ 0 255 102} { 0 255 0}
		{ 102 255 0} { 204 255 0}
		{ 255 234 0} { 255 204 0}
		{ 255 102 0} { 255 0 0}}}
	} elseif {$which == "Blackbody"} {
	    set color { "Blackbody" {	
		{0 0 0}   {52 0 0}
		{102 2 0}   {153 18 0}
		{200 41 0}   {230 71 0}
		{255 120 0}   {255 163 20}
		{255 204 55}   {255 228 80}
		{255 247 120}   {255 255 180}
		{255 255 255}}}
	} elseif {$which == "Darkhue"} {
	    set color { "Darkhue" {	
		{ 0  0  0 }  { 0 28 39 }
		{ 0 30 55 }  { 0 15 74 }
		{ 1  0 76 }  { 28  0 84 }
		{ 32  0 85 }  { 57  1 92 }
		{ 108  0 114 }  { 135  0 105 }
		{ 158  1 72 }  { 177  1 39 }
		{ 220  10 10 }  { 229 30  1 }
		{ 246 72  1 }  { 255 175 36 }
		{ 255 231 68 }  { 251 255 121 }
		{ 239 253 174 }}}
	} elseif {$which == "Red-to-Blue"} {
	    set color { "Red-to-Blue" { { 0 0 255 } { 255 255 255} { 255 0 0 } } }
	}

        set colorMap [$this set_color_map $color]
	
	set width $colormap_width
        set height $colormap_height
	
	set n [llength $colorMap]
	$canvas delete map
	set dx [expr $width/double($n)] 
	set x 0
	for {set i 0} {$i < $n} {incr i 1} {
	    set color [lindex $colorMap $i]
	    set r [lindex $color 0]
	    set g [lindex $color 1]
	    set b [lindex $color 2]
	    set c [format "#%02x%02x%02x" $r $g $b]
	    set oldx $x
	    set x [expr ($i+1)*$dx]
	    $canvas create rectangle \
		$oldx 0 $x $height -fill $c -outline $c -tags map
	}
    }

    
    ######################
    ### set_color_map
    ######################
    method set_color_map { map } {
        set resolution $colormap_res
	set colorMap {}
	set currentMap {}
	set currentMap [$this make_new_map [ lindex $map 1 ]]
	set n [llength $currentMap]
	if { $resolution < $n } {
	    set resolution $n
	}
	set m $resolution
	
	set frac [expr ($n-1)/double($m-1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
		lappend color 0.5
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
		lappend color 0.5
	    } else {
		set index [expr int($i * $frac)]
		set t [expr ($i * $frac)-$index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
		lappend color 0.5
	    }
	    lappend colorMap $color
	}	
        return $colorMap
    }
    
    

    #######################
    ### make_new_map
    #######################
    method make_new_map { currentMap } {
        set gamma 0
	set res $colormap_res
	set newMap {}
	set m [expr int($res + abs( $gamma )*(255 - $res))]
	set n [llength $currentMap]
	if { $m < $n } { set m $n }
	set frac [expr double($n-1)/double($m - 1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
	    } else {
		set index_double [$this modify [expr $i * $frac] [expr $n-1]]
		
		set index [expr int($index_double)]
		set t  [expr $index_double - $index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
	    }
	    lappend newMap $color
	}
	return $newMap
    }
    

    #####################
    ### modify
    #####################
    method modify { i range } {
	set gamma 0
	
	set val [expr $i/double($range)]
	set bp [expr tan( 1.570796327*(0.5 + $gamma*0.49999))]
	set index [expr pow($val,$bp)]
	return $index*$range
    }


    #######################
    ### addColorSelection
    #######################
    # This method creates a button and color swatch so the
    # user can set a specific color
    method addColorSelection {frame text color mod} {
	#add node color picking 
	global $color
	global $color-r
	global $color-g
	global $color-b
	#add node color picking 
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]
	
	frame $frame.colorFrame
	frame $frame.colorFrame.col -relief ridge -borderwidth \
	    4 -height 25 -width 25 \
	    -background [format #%04x%04x%04x $ir $ig $ib]
			 
	set cmmd "$this raiseColor $frame.colorFrame.col $color $mod"
	button $frame.colorFrame.set_color \
	    -state disabled \
	    -text $text -command $cmmd
	
	#pack the node color frame
	pack $frame.colorFrame.set_color \
	    -side left -ipadx 2 -ipady 2
	pack $frame.colorFrame.col -side left 
	pack $frame.colorFrame -side left -padx 1
    }
    
    
    #######################
    ### raiseColor
    #######################
    # Raises the color swatch and allows the user to pick
    method raiseColor {col color mod} {
	global $color
	set window .standalone
	if {[winfo exists $window.color]} {
	    raise $window.color
	    return;
	} else {
	    toplevel $window.color
	    makeColorPicker $window.color $color \
		"$this setColor $col $color $mod" \
		"destroy $window.color"
	}
    }

	
    #########################
    ### setColor
    #########################
    # Set the appropriate color.  This should probably be
    # re-written by the child class to change the color
    # for a specific ShowField.
    method setColor {col color mode} {
	global $color
	global $color-r
	global $color-g
	global $color-b
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]
	
	set window .standalone
	$col config -background [format #%04x%04x%04x $ir $ig $ib]
    }
	

    

    ###############################
    ### Class Variables
    ###############################

    # Embedded Viewer
    variable eviewer

    # Standalone
    variable win
    
    # The width and height of the left frame, or the processing frame. 
    # This will be used when attaching or detaching the frame.
    variable process_width
    variable process_height

    # The width and height of the Viewer window. This is used when configuring
    # the window size when frames are attached or detached.
    variable viewer_width
    variable viewer_height

    # The width and height of the right frame, or the visualization frame. 
    # This will be used when attaching or detaching the frame.
    variable vis_width
    variable vis_height

    # The width and height of the current screen so that the app can 
    # come up in the center
    variable screen_width
    variable screen_height

    # Indicates whether the processing frame is attached (1) or detached (0).
    # Applications should be initialized as attached
    variable IsPAttached

    # Pointers to the attached and detached frames
    variable detachedPFr
    variable attachedPFr

    # Indicates whether the visualization frame is attached (1) or detached (0).
    # Applications should be initialized as attached
    variable IsVAttached

    # Pointers to the attached and detached frames
    variable detachedVFr
    variable attachedVFr

    # Flag to indicate whether entire gui has been built.
    # This is usefull in functions called by scale widgets because they
    # get called during initialization.
    variable initialized

    # Flag to indicate when app is loading and executing a saved session.
    variable loading

    # The id for the call to change_indicator which controls the progress bar.
    variable indicatorID
    
    # These point to the indicator canvas and are used to raise different canvases.
    # One points to the attached frame and the other to the detached
    variable indicator1
    variable indicator2

    # These point to the progress indicator labels of the attached and detached frames
    variable indicatorL1
    variable indicatorL2

    # Represents the state of the progress graph
    # 0 = Reset the indicator
    # 1 = Start executing spinner
    # 2 = Stop executing spinner and indicate "Complete"
    # 3 = Error state
    variable indicate

    # The progress spinner is just an image shifting upon each executing.  The cyle
    # variable keeps track of how many shifts the image has made.
    variable cycle

    # A counter of currently executing modules.  If this is positive, then the progress indicator
    # should be spinning.  Only when all modules are complete (and this variable is equal to 0) will
    # the "Complete" state be reached.
    variable executing_modules

    # Width and height of the indicator
    variable i_width
    variable i_height
    
    # Number of stripes on the spinner mode of the indicator
    variable stripes

    # Variables for positions when moving spinner image
    variable i_move
    variable i_back

    # Stores name of first module to get an error and the indicator won't
    # be cleared from the error state until this module clears it.
    variable error_module

    # Colors for execute and next buttons
    variable next_color
    variable execute_color

    # Width and height and resolution of colormap canvases if needed
    variable colormap_width
    variable colormap_height
    variable colormap_res    
    
}
