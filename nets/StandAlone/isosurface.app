# SCI Network 1.0

package require Iwidgets 3.0

::netedit dontschedule

global notes
set notes ""

# global array indexed by module name to keep track of modules
global mods

global standalone_progress

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
	set notebook_height 2.6i

	set iso_alg 1
    }

    destructor {
	destroy $this
    }

    method BuildApp {} {
	global mods

	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow .standalone.viewer

	# pack viewer
	pack $win.viewer -side left -anchor nw 

	### Vis Part
	#####################
	### Create a Detached Vis Part
	toplevel $win.detachedV
	frame $win.detachedV.f -relief flat
	pack $win.detachedV.f -side left -anchor nw

	wm title $win.detachedV "Visualization Window"

	wm sizefrom $win.detachedV user
	wm positionfrom $win.detachedV user
	
	wm withdraw $win.detachedV

	### Create Attached Vis Part
	frame $win.attachedV
	frame $win.attachedV.f -relief flat
	pack $win.attachedV.f -side left -anchor nw -fill y

	set IsVAttached 1

	set att_msg "Detach from Viewer"
	set det_msg " Attach to Viewer "

	# set frame data members
	set detachedVFr $win.detachedV
	set attachedVFr $win.attachedV
	
	init_Vframe $detachedVFr.f $det_msg 0
	init_Vframe $attachedVFr.f $att_msg 1

	# show the process frame
	pack $attachedVFr -side left -anchor nw -after $win.viewer -fill y
#	append geom [expr [winfo width $win]>[winfo width $win.attachedV] ?[winfo width $win]:[winfo width $win.attachedV]] x [expr [winfo height $win]+[winfo reqheight $win.attachedV]]
	#wm geometry $win $geom

	### Processing Part
	#########################
	### Create Detached Processing Part
	toplevel $win.detachedP
	frame $win.detachedP.f -relief flat
	pack $win.detachedP.f -side left -anchor nw

	wm title $win.detachedP "Processing Window"
	
	wm sizefrom $win.detachedP user
	wm positionfrom $win.detachedP user

	wm withdraw $win.detachedP


	### Create Attached Processing Part
	frame $win.attachedP 
	frame $win.attachedP.f -relief flat 
	pack $win.attachedP.f -side left -anchor nw -fill y

	set IsPAttached 1

	# set frame data members
	set detachedPFr $win.detachedP
	set attachedPFr $win.attachedP

	init_Pframe $detachedPFr.f $det_msg
	init_Pframe $attachedPFr.f $att_msg

	# show the process frame
	pack $attachedPFr -side left -anchor nw -before $win.viewer -fill y
#	append geom [expr [winfo width $win]>[winfo width $win.attachedP] ?[winfo width $win]:[winfo width $win.attachedP]] x [expr [winfo height $win]+[winfo reqheight $win.attachedP]]
	#wm geometry $win $geom
	update

    }


    method init_Pframe { m msg } {

	if { [winfo exists $m] } {
	    ### Menu
	    frame $m.main_menu -relief raised -borderwidth 3
	    pack $m.main_menu -fill x
	    
	    menubutton $m.main_menu.file -text "File" -underline 0 \
		-menu $m.main_menu.file.menu
	    
	    menu $m.main_menu.file.menu -tearoff false
	    
	    $m.main_menu.file.menu add command -label "Save Ctr+S" \
		-underline 0 -command "$this SaveSession" -state active
	    
	    $m.main_menu.file.menu add command -label "Load  Ctr+L" \
		-underline 0 -command "$this LoadSession" -state active
	    
	    $m.main_menu.file.menu add command -label "Quit   Ctr+Q" \
		-underline 0 -command "$this ExitApp" -state active
	    
	    pack $m.main_menu.file -side left
	    
	    menubutton $m.main_menu.help -text "Help" -underline 0 \
		-menu $m.main_menu.help.menu
	    
	    menu $m.main_menu.help.menu -tearoff false
	    
	    $m.main_menu.help.menu add command -label "View Help" \
		-underline 0 -command "$this ShowHelp" -state active
	    
	    pack $m.main_menu.help -side left
	    
	    tk_menuBar $m.main_menu $win.main_menu.file $win.main_menu.help
	    
	    ### Processing Steps
	    #####################
	    iwidgets::labeledframe $m.p \
		-labelpos n -labeltext "Processing Steps"
	    pack $m.p -side top -fill y -anchor nw
	    
	    set process [$m.p childsite]
	    
	    ### Data Section
	    iwidgets::labeledframe $process.data \
		-labelpos nw -labeltext "Data" 
	    pack $process.data -side top -fill y -anchor nw
	    
	    set data_section [$process.data childsite]
	    
	    message $data_section.message -width 200 \
		-text "Please load a dataset to isosurface."
	    pack $data_section.message -side top  -anchor n
	    button $data_section.load -text "Load" -command "$this LoadData"  \
		-width 15 
	    pack $data_section.load -side top -padx 2 -pady 5  -ipadx 3 -ipady 3 -anchor n
	    
	    ### Progress
	    iwidgets::labeledframe $process.progress \
		-labelpos nw -labeltext "Progress" 
	    pack $process.progress -side top -anchor nw -fill x
	    
	    set progress_section [$process.progress childsite]
	    iwidgets::feedback $progress_section.fb -labeltext "Isosurfacing..." \
		-steps 20 -barcolor Green \
		
	    pack $progress_section.fb -side top -padx 2 -pady 2 -anchor nw -fill x
	    
	    global standalone_progress
	    set standalone_progress $progress_section.fb

	    ### Attach/Detach button
	    button $process.cut -text $msg -command "$this switch_P_frames"
	    pack $process.cut -side top -anchor s -pady 5 -padx 5 
	}
	    
    }

    method init_Vframe { m msg case} {
	global mods
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization"
	    pack $m.vis -side top -anchor nw 
	    
	    set vis [$m.vis childsite]
	    
	    iwidgets::scrolledlistbox $vis.data -labeltext "Loaded Data" \
		-vscrollmode dynamic -hscrollmode dynamic \
		-selectmode single \
		-height 0.9i \
		-width $notebook_width \
		-labelpos nw -selectioncommand "$this SelectVisData"
	    
	    pack $vis.data -padx 4 -pady 4
	    
	    if {$case == 0} {
		# detached case
		set data_listbox_Det $vis.data
	    } else {
		# attached case
		set data_listbox_Att $vis.data
	    }
	    
	    $vis.data insert 0 {None}
	    
	    
	    ### Data Info
	    frame $vis.f -relief groove -borderwidth 2
	    pack $vis.f -side top -anchor nw -fill x
	    
	    iwidgets::notebook $vis.f.nb -width $notebook_width \
		-height $notebook_height 
	    pack $vis.f.nb -padx 4 -pady 4

	    if {$case == 0} {
		# detached case
		set notebook_Det $vis.f.nb
	    } else {
		# attached case
		set notebook_Att $vis.f.nb
	    }
	    
	    ### Renderer Options
	    iwidgets::labeledframe $vis.viewer_opts \
		-labelpos nw -labeltext "Global Render Options"
	    
	    pack $vis.viewer_opts -side top -anchor nw -fill x
	    
	    set view_opts [$vis.viewer_opts childsite]

	    checkbutton $view_opts.light -text "Lighting" \
		-variable $mods(Viewer)-ViewWindow_0-global-light \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"

	    checkbutton $view_opts.fog -text "Fog" \
		-variable $mods(Viewer)-ViewWindow_0-global-fog \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"

	    checkbutton $view_opts.bbox -text "BBox" \
		-variable $mods(Viewer)-ViewWindow_0-global-debug \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"

	    checkbutton $view_opts.clip -text "Use Clip" \
		-variable $mods(Viewer)-ViewWindow_0-global-clip \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"

	    checkbutton $view_opts.cull -text "Back Cull" \
		-variable $mods(Viewer)-ViewWindow_0-global-cull \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"

	    checkbutton $view_opts.dl -text "Display List" \
		-variable $mods(Viewer)-ViewWindow_0-global-dl \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"


	    pack $view_opts.light $view_opts.fog $view_opts.bbox \
		$view_opts.clip $view_opts.cull $view_opts.dl \
		-side top -anchor w

	    button $view_opts.autoview -text "Autoview" \
		-command {$m3-ViewWindow_0-c autoview} \
		-width 20
	    pack $view_opts.autoview -side top -padx 3 -pady 3 -anchor n

	    ### Attach/Detach button
	    button $vis.cut -text $msg -command "$this switch_V_frames"
	    pack $vis.cut -side top -anchor s -pady 5 -padx 5 

	}
    }


    method switch_P_frames {} {
	if { $IsPAttached } {	    
	    pack forget $attachedPFr
	    #append geom [winfo width $win] x [expr [winfo height $win]-[winfo reqheight $win.attachedP]]
	    wm geometry $detachedPFr "198x512+0+0"
	    wm deiconify $detachedPFr
	    set IsPAttached 0
	} else {
	    wm withdraw $detachedPFr
	    pack $attachedPFr -anchor nw -side left -before $win.viewer
	    #append geom [winfo width $win] x [expr [winfo height $win]+[winfo reqheight $win.attachedP]]
	    #wm geometry $win $geom
	    set IsPAttached 1
	}
	update
    }

    method switch_V_frames {} {
	if { $IsVAttached } {
	    # select data in detached data list box
	    # FIX ME (check if exists before getting index)
	    $data_listbox_Det selection set [$data_listbox_Att curselection] [$data_listbox_Att curselection]

	    pack forget $attachedVFr
	    #append geom [winfo width $win] x [expr [winfo height $win]-[winfo reqheight $win.attachedV]]
	    wm geometry $detachedVFr "300x512+0+0"
	    wm deiconify $detachedVFr
	    set IsVAttached 0
	} else {
	    # select data in attached data list box
	    # FIX ME (check if exists before getting index)
	    $data_listbox_Att selection set [$data_listbox_Det curselection] [$data_listbox_Det curselection]

	    wm withdraw $detachedVFr
	    pack $attachedVFr -anchor nw -side left -after $win.viewer
	    #append geom [winfo width $win] x [expr [winfo height $win]+[winfo reqheight $win.attachedV]]
	    #wm geometry $win $geom
	    set IsVAttached 1
	}
	update
    }

    method LoadData {} {
	# Bring up FieldReader UI
	global mods
	global $mods(FieldReader)-filename
	$mods(FieldReader) ui
	
	tkwait window .ui$mods(FieldReader)
	
	# get the filename and stick into an array formated 
	# data(file) = full path
	set name [GetFileName [set $mods(FieldReader)-filename] ]
	set data($name) [set $mods(FieldReader)-filename]
	
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
	BuildDataInfoPage $notebook_Att $name
	BuildDataInfoPage $notebook_Det $name

	# bring new data info page forward
	$notebook_Att view $name
	$notebook_Det view $name
	
	# reset progress
	global standalone_progress
	$standalone_progress reset

    }

    method SelectVisData {} {
	global mods
	global $mods(FieldReader)-filename
	set current ""
	if {$IsVAttached} {
	    set current [$data_listbox_Att getcurselection]
	} else {
	    set current [$data_listbox_Det getcurselection]
	}
	puts "SelectVisData current: $current"
	if {[info exists data($current)] == 1} {

            # bring data info page forward
	    $notebook_Att view $current
	    $notebook_Det view $current
	    
	    # data selected - update FieldReader and execute
	    set $mods(FieldReader)-filename $data($current)
	    $mods(FieldReader)-c needexecute
	} 
	#else {
	#    set result [tk_messageBox -parent . \
	    \#	    -title {No Data} -type ok \
	    \#    -icon error \
	    \#  -message "Please load select a valid data set."]
    #}
    
   }

    method SaveSession {} {
	puts "NEED TO IMPLEMENT SAVE SESSION"
    }
    
    method LoadSession {} {
	puts "NEED TO IMPLEMENT LOAD SESSION"
    }
        
    method ExitApp {} {
	netedit quit
    }

    method ShowHelp {} {
	puts "NEED TO IMPLEMENT SHOW HELP"
    }

    method UpdateIsosurfaceScale {} {
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

    method UpdateIsosurface { isoval } {

	global mods

	# execute with the given value
	[$mods(Isosurface) get_this_c] needexecute

	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

    }
    
    method GetFileName { filename } {
	set end [string length $filename]
	set start [string last "/" $filename]
	set start [expr 1 + $start]
	
	return [string range $filename $start $end]	
    }

    method BuildDataInfoPage { w which } {
	set page [$w add -label $which]

	iwidgets::scrolledframe $page.sf \
	    -width $notebook_width -height $notebook_height \
	    -labeltext "Data: $which" \
	    -vscrollmode none \
	    -hscrollmode dynamic \
	    -background Grey

	pack $page.sf -anchor n -fill x

	AddIsosurface [$page.sf childsite] $which

    }

    method AddIsosurface { w which } {
	global mods
	global $mods(Isosurface)-isoval
	global $mods(Isosurface)-isoval-min
	global $mods(Isosurface)-isoval-max

	iwidgets::labeledframe $w.isosurface \
	    -labelpos nw -labeltext "Isosurface"

	pack $w.isosurface -side top -anchor n -fill x -fill y

	set isosurface [$w.isosurface childsite]

	# isosurface sliders
	scale $isosurface.isoval -label "Isoval:" \
 	    -variable $mods(Isosurface)-isoval \
 	    -from [set $mods(Isosurface)-isoval-min] \
 	    -to [set $mods(Isosurface)-isoval-max] \
 	    -length 5c \
 	    -showvalue true \
 	    -orient horizontal \
 	    -digits 5 \
 	    -resolution 0.001 \
 	    -command "$this UpdateIsosurface" \
	
 	pack $isosurface.isoval -side top 

	# isosurface method
	label $isosurface.mlabel -text "Method"
	pack $isosurface.mlabel -side top -anchor nw 

	frame $isosurface.method -relief flat
	pack $isosurface.method -side top -anchor nw -fill x

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
	    -anchor nw -fill x

	# turn on Noise by default
	$isosurface.method.noise invoke

	
	button $isosurface.adv -text "Advanced" -width 20 \
	    -command "$mods(Isosurface) ui"
	pack $isosurface.adv -side top -padx 4 -pady 4 -anchor n
	
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
    variable IsPAttached
    variable detachedPFr
    variable attachedPFr
    variable IsVAttached
    variable detachedVFr
    variable attachedVFr
}

IsoApp app

app BuildApp






### Bind shortcuts - Must be after instantiation of IsoApp
bind .standalone <Control-s> {
    app SaveSession
}

bind .standalone <Control-l> {
    app LoadSession
}

bind .standalone <Control-q> {
    app ExitApp
}






