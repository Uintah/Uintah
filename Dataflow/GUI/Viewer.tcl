#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

#catch {rename Viewer ""} 

itcl_class SCIRun_Render_Viewer {
    inherit Module

    # List of ViewWindow children of this Viewer
    protected viewwindow

    # Id for the Next ViewWindows to be created.  Incremented for each new Viewindow
    protected nextrid 0

    constructor {config} {
	set name Viewer
	set_defaults
    }
    destructor {
	foreach rid $viewwindow {
	    destroy .ui[$rid modname]

	    $rid delete
	}
    }

    method set_defaults {} {
	set make_progress_graph 0
	set make_time 0
	set viewwindow ""
    }

    method makeViewWindowID {} {
	set id $this-ViewWindow_$nextrid
	incr nextrid
	while {[::info commands $id] != ""} {
	    set id $this-ViewWindow_$nextrid
	    incr nextrid
	}
	return $id
    }

    method ui {{rid -1}} {
	if {$rid == -1} {
	    set rid [makeViewWindowID]
	}
        
        ViewWindow $rid -viewer $this 

	lappend viewwindow $rid
    }

    method ui_embedded {{rid -1}} {
	if {$rid == -1} {
	    set rid [makeViewWindowID]
	}
        
        set result [EmbeddedViewWindow $rid -viewer $this]

	lappend viewwindow $rid
	return $result
    }
}

catch {rename ViewWindow ""}

itcl_class ViewWindow {
    public viewer
    
    # parameters to hold current state of detachable part
    protected IsAttached 
    protected IsDisplayed
    # hold names of detached and attached windows
    protected detachedFr
    protected attachedFr

    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }

    method set_defaults {} {

	# set defaults values for parameters that weren't set in a script

        # CollabVis code begin 
        global $this-have_collab_vis 
        # CollabVis code end

	global $this-saveFile
	global $this-saveType
	if {![info exists $this-File]} {set $this-saveFile "out.raw"}
	if {![info exists $this-saveType]} {set $this-saveType "raw"}

	# Animation parameters
	global $this-current_time
	if {![info exists $this-current_time]} {set $this-current_time 0}
	global $this-tbeg
	if {![info exists $this-tbeg]} {set $this-tbeg 0}
	global $this-tend
	if {![info exists $this-tend]} {set $this-tend 1}
	global $this-framerate
	if {![info exists $this-framerate]} {set $this-framerate 15}
	global $this-totframes
	if {![info exists $this-totframes]} {set $this-totframes 30}
	global $this-caxes
        if {![info exists $this-caxes]} {set $this-caxes 0}
	global $this-raxes
        if {![info exists $this-raxes]} {set $this-raxes 1}

	# Need to initialize the background color
	global $this-bgcolor-r
	if {![info exists $this-bgcolor-r]} {set $this-bgcolor-r 0}
	global $this-bgcolor-g
	if {![info exists $this-bgcolor-g]} {set $this-bgcolor-g 0}
	global $this-bgcolor-b
	if {![info exists $this-bgcolor-b]} {set $this-bgcolor-b 0}

	# Need to initialize the scene material scales
	global $this-ambient-scale
	if {![info exists $this-ambient-scale]} {set $this-ambient-scale 1.0}
	global $this-diffuse-scale
	if {![info exists $this-diffuse-scale]} {set $this-diffuse-scale 1.0}
	global $this-specular-scale
	if {![info exists $this-specular-scale]} {set $this-specular-scale 0.4}
	global $this-emission-scale
	if {![info exists $this-emission-scale]} {set $this-emission-scale 1.0}
	global $this-shininess-scale
	if {![info exists $this-shininess-scale]} {set $this-shininess-scale 1.0}
	# Initialize point size, line width, and polygon offset
	global $this-point-size
	if {![info exists $this-point-size]} {set $this-point-size 1.0}
	global $this-line-width
	if {![info exists $this-line-width]} {set $this-line-width 1.0}
	global $this-polygon-offset-factor
 	if {![info exists $this-polygon-offset-factor]} \
	    {set $this-polygon-offset-factor 1.0}
	global $this-polygon-offset-units
	if {![info exists $this-polygon-offset-units]} \
	    {set $this-polygon-offset-units 0.0}

	# Set up lights
	global $this-global-light0 # light 0 is the head light
	if {![info exists $this-global-light0]} { set $this-global-light0 1 }
	global $this-global-light1 
	if {![info exists $this-global-light1]} { set $this-global-light1 0 }
	global $this-global-light2 
	if {![info exists $this-global-light2]} { set $this-global-light2 0 }
	global $this-global-light3 
	if {![info exists $this-global-light3]} { set $this-global-light3 0 }
# 	global $this-global-light4 
# 	if {![info exists $this-global-light4]} { set $this-global-light4 0 }
# 	global $this-global-light5 
# 	if {![info exists $this-global-light5]} { set $this-global-light5 0 }
# 	global $this-global-light6
# 	if {![info exists $this-global-light6]} { set $this-global-light6 0 }
# 	global $this-global-light7 
# 	if {![info exists $this-global-light7]} { set $this-global-light7 0 }
	global $this-lightVectors
	if {![info exists $this-lightVectors]} { 
	    set $this-lightVectors \
		[list { 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }]
# 		     { 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }]
	}
	if {![info exists $this-lightColors]} {
	    set $this-lightColors \
		[list {1.0 1.0 1.0} {1.0 1.0 1.0} \
		     {1.0 1.0 1.0} {1.0 1.0 1.0} ]
# 		     {1.0 1.0 1.0} {1.0 1.0 1.0} \
# 		     {1.0 1.0 1.0} {1.0 1.0 1.0} ]
	}

	global $this-sbase
	if {![info exists $this-sbase]} {set $this-sbase 0.4}
	global $this-sr
	if {![info exists $this-sr]} {set $this-sr 1}
	global $this-do_stereo
	if {![info exists $this-do_stereo]} {set $this-do_stereo 0}

	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	set $this-def-color-r 1.0
	set $this-def-color-g 1.0
	set $this-def-color-b 1.0

        # CollabVis code begin
        if {[set $this-have_collab_vis]} {
	    global $this-view_server
	    if {![info exists $this-view_server]} {set $this-view_server 0}
        }
        # CollabVis code end

	global $this-ortho-view
	if {![info exists $this-ortho-view]} { set $this-ortho-view 0 }
    }

    destructor {
    }

    constructor {config} {
	$viewer-c addviewwindow $this
	set w .ui[modname]
	toplevel $w
	bind $w <Destroy> "$this killWindow %W" 
	wm title $w "ViewWindow"
	wm iconname $w "ViewWindow"
	wm minsize $w 100 100
	set_defaults 

	frame $w.menu -relief raised -borderwidth 3
	pack $w.menu -fill x

	menubutton $w.menu.file -text "File" -underline 0 \
		-menu $w.menu.file.menu
	menu $w.menu.file.menu
#	$w.menu.file.menu add command -label "Save geom file..." -underline 0 \
#		-command "$this makeSaveObjectsPopup"
	$w.menu.file.menu add command -label "Save image file..." \
	    -underline 0 -command "$this makeSaveImagePopup"


	# Get the list of supported renderers for the pulldown
	
	frame $w.wframe -borderwidth 3 -relief sunken
	pack $w.wframe -expand yes -fill both -padx 4 -pady 4

	set width 640
	set height 512

	menubutton $w.menu.edit -text "Edit" -underline 0 \
		-menu $w.menu.edit.menu
	menu $w.menu.edit.menu
	$w.menu.edit.menu add command -label "View/Camera..." -underline 0 \
		-command "$this makeViewPopup"
#	$w.menu.edit.menu add command -label "Renderer..." -underline 0
#	$w.menu.edit.menu add command -label "Materials..." -underline 0
	$w.menu.edit.menu add command -label "Light Sources..." -underline 0 \
	    -command "$this makeLightSources"
	$w.menu.edit.menu add command -label "Background..." -underline 0 \
		-command "$this makeBackgroundPopup"
	$w.menu.edit.menu add command -label "Clipping Planes..." -underline 0 -command "$this makeClipPopup"
#	$w.menu.edit.menu add command -label "Animation..." -underline 0 \
#		-command "$this makeAnimationPopup"
	$w.menu.edit.menu add command -label "Point Size..." -underline 0 \
		-command "$this makePointSizePopup"
	$w.menu.edit.menu add command -label "Line Width..." -underline 0 \
		-command "$this makeLineWidthPopup"
	$w.menu.edit.menu add command -label "Polygon Offset..." -underline 0 \
		-command "$this makePolygonOffsetPopup"
	$w.menu.edit.menu add command -label "Scene Materials..." -underline 0 \
		-command "$this makeSceneMaterialsPopup"
#	menubutton $w.menu.spawn -text "Spawn" -underline 0 \
#		-menu $w.menu.spawn.menu
#	menu $w.menu.spawn.menu
#	$w.menu.spawn.menu add command -label "Spawn Independent..." -underline 6
#	$w.menu.spawn.menu add command -label "Spawn Child..." -underline 6
#	menubutton $w.menu.dialbox -text "Dialbox" -underline 0 \
#		-menu $w.menu.dialbox.menu
#	menu $w.menu.dialbox.menu
#	$w.menu.dialbox.menu add command -label "Translate/Scale..." -underline 0 \
#		-command "$w.dialbox connect"
#	$w.menu.dialbox.menu add command -label "Camera..." -underline 0 \
#		-command "$w.dialbox2 connect"

	menubutton $w.menu.visual -text "Visual" -underline 0 \
	    -menu $w.menu.visual.menu
	menu $w.menu.visual.menu
	set i 0
	global $this-currentvisual
	set $this-currentvisual 0
	foreach t [$this-c listvisuals $w] {
	    $w.menu.visual.menu add radiobutton -value $i -label $t \
		-variable $this-currentvisual \
		-font "-Adobe-Helvetica-bold-R-Normal-*-12-75-*" \
		-command "$this switchvisual $i"
#        -command { puts "switchvisual doesn't work on NT" }
#puts "$i: $t"
	    incr i
	}

	pack $w.menu.file -side left
	pack $w.menu.edit -side left
#	pack $w.menu.renderer -side left
#	pack $w.menu.spawn -side left
#	pack $w.menu.dialbox -side left
	pack $w.menu.visual -side left
#	tk_menuBar $w.menu $w.menu.edit $w.menu.renderer \
#		$w.menu.spawn $w.menu.dialbox $w.menu.visual

	# Create Dialbox and attach to it
	Dialbox $w.dialbox "Viewer - Translate/Scale"
	$w.dialbox unbounded_dial 0 "Translate X" 0.0 1.0 "$this translate x"
	$w.dialbox unbounded_dial 2 "Translate Y" 0.0 1.0 "$this translate y" 
	$w.dialbox unbounded_dial 4 "Translate Z" 0.0 1.0 "$this translate z"
	$w.dialbox wrapped_dial 1 "Rotate X" 0.0 0.0 360.0 1.0 "$this rotate x"
	$w.dialbox wrapped_dial 3 "Rotate Y" 0.0 0.0 360.0 1.0 "$this rotate y"
	$w.dialbox wrapped_dial 5 "Rotate Z" 0.0 0.0 360.0 1.0 "$this rotate z"
	$w.dialbox bounded_dial 6 "Scale" 1.0 [expr 1.0/1000.0] 1000.0 1.0 "$this scale"
	
	# Create Dialbox2 and attach to it
	Dialbox $w.dialbox2 "Viewer - Camera"
	$w.dialbox2 bounded_dial 0 "Zoom" 0.0 0.0 1000.0 100.0 "$this zoom"
	$w.dialbox2 wrapped_dial 1 "Pan" 0.0 0.0 360.0 1.0 "$this pan" 
	$w.dialbox2 wrapped_dial 2 "Tilt" 0.0 0.0 360.0 1.0 "$this tilt"
	$w.dialbox2 bounded_dial 3 "FOV" 0.0 0.0 180.0 1.0 "$this fov"

	frame $w.bframe
	pack $w.bframe -side top -fill x
	frame $w.bframe.pf
	pack $w.bframe.pf -side left -anchor n
	label $w.bframe.pf.perf1 -width 32 -text "100000 polygons in 12.33 seconds"
	pack $w.bframe.pf.perf1 -side top -anchor n
	label $w.bframe.pf.perf2 -width 32 -text "Hello"
	pack $w.bframe.pf.perf2 -side top -anchor n
	label $w.bframe.pf.perf3 -width 32 -text "Hello"
	pack $w.bframe.pf.perf3 -side top -anchor n

	canvas $w.bframe.mousemode -width 200 -height 70 \
		-relief groove -borderwidth 2
	pack $w.bframe.mousemode -side left -fill y -pady 2 -padx 2
	global $w.bframe.mousemode.text
	set mouseModeText $w.bframe.mousemode.text
	$w.bframe.mousemode create text 35 40 -tag mouseModeText \
		-text " Current Mouse Mode " \
		-anchor w

	frame $w.bframe.v1
	pack $w.bframe.v1 -side left
	button $w.bframe.v1.autoview -text "Autoview" -command "$this-c autoview"
	pack $w.bframe.v1.autoview -fill x -pady 2 -padx 2

	frame $w.bframe.v1.views             
	pack $w.bframe.v1.views -side left -anchor nw -fill x -expand 1
	
	menubutton $w.bframe.v1.views.def -text "   Views   " -menu $w.bframe.v1.views.def.m -relief raised -padx 2 -pady 2
	
	menu $w.bframe.v1.views.def.m
	$w.bframe.v1.views.def.m add cascade -label "Look down +X Axis" \
		-menu $w.bframe.v1.views.def.m.posx
	$w.bframe.v1.views.def.m add cascade -label "Look down +Y Axis" \
		-menu $w.bframe.v1.views.def.m.posy
        $w.bframe.v1.views.def.m add cascade -label "Look down +Z Axis" \
		-menu $w.bframe.v1.views.def.m.posz
	$w.bframe.v1.views.def.m add separator
	$w.bframe.v1.views.def.m add cascade -label "Look down -X Axis" \
		-menu $w.bframe.v1.views.def.m.negx
	$w.bframe.v1.views.def.m add cascade -label "Look down -Y Axis" \
		-menu $w.bframe.v1.views.def.m.negy
        $w.bframe.v1.views.def.m add cascade -label "Look down -Z Axis" \
		-menu $w.bframe.v1.views.def.m.negz

	pack $w.bframe.v1.views.def -side left -pady 2 -padx 2 -fill x

	menu $w.bframe.v1.views.def.m.posx
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector +Y" \
		-variable $this-pos -value x1_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector -Y" \
		-variable $this-pos -value x1_y0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector +Z" \
		-variable $this-pos -value x1_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector -Z" \
		-variable $this-pos -value x1_z0 -command "$this-c Views"

	menu $w.bframe.v1.views.def.m.posy
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector +X" \
		-variable $this-pos -value y1_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector -X" \
		-variable $this-pos -value y1_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector +Z" \
		-variable $this-pos -value y1_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector -Z" \
		-variable $this-pos -value y1_z0 -command "$this-c Views"

	menu $w.bframe.v1.views.def.m.posz
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector +X" \
		-variable $this-pos -value z1_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector -X" \
		-variable $this-pos -value z1_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector +Y" \
		-variable $this-pos -value z1_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector -Y" \
		-variable $this-pos -value z1_y0 -command "$this-c Views"

	menu $w.bframe.v1.views.def.m.negx
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector +Y" \
		-variable $this-pos -value x0_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector -Y" \
		-variable $this-pos -value x0_y0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector +Z" \
		-variable $this-pos -value x0_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector -Z" \
		-variable $this-pos -value x0_z0 -command "$this-c Views"

	menu $w.bframe.v1.views.def.m.negy
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector +X" \
		-variable $this-pos -value y0_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector -X" \
		-variable $this-pos -value y0_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector +Z" \
		-variable $this-pos -value y0_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector -Z" \
		-variable $this-pos -value y0_z0 -command "$this-c Views"

	menu $w.bframe.v1.views.def.m.negz
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector +X" \
		-variable $this-pos -value z0_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector -X" \
		-variable $this-pos -value z0_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector +Y" \
		-variable $this-pos -value z0_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector -Y" \
		-variable $this-pos -value z0_y0 -command "$this-c Views"

	frame $w.bframe.v2
	pack $w.bframe.v2 -side left -padx 2 -pady 2
	button $w.bframe.v2.sethome -text "Set Home View" -padx 2 \
		-command "$this-c sethome"
	pack $w.bframe.v2.sethome -fill x -pady 2
	button $w.bframe.v2.gohome -text "Go home" -command "$this-c gohome"
	pack $w.bframe.v2.gohome -fill x -pady 2
	
	button $w.bframe.more -text "+" -padx 3 \
		-font "-Adobe-Helvetica-bold-R-Normal-*-12-75-*" \
		-command "$this addMFrame $w"
	pack $w.bframe.more -pady 2 -padx 2 -anchor se -side right

# AS: initialization of attachment
	toplevel $w.detached
	frame $w.detached.f
	pack $w.detached.f -side top -anchor w -fill x
	
	wm title $w.detached "VIEWWINDOW settings"
	# This update messes up SCIRun2 - is it necessary? Steve
	#update

	wm sizefrom  $w.detached user
	wm positionfrom  $w.detached user

	wm protocol $w.detached WM_DELETE_WINDOW "$this removeMFrame $w"
	wm withdraw $w.detached
	
	frame $w.mframe
	frame $w.mframe.f
	pack $w.mframe.f -side top -fill x

	set IsAttached 1
	set IsDisplayed 0
	
	set att_msg "Double-click here to detach - - - - - - - - - - - - - - - - - - - - -"
	set det_msg "Double-click here to attach - - - - - - - - - - - - - - - - - - - - -"
	set detachedFr $w.detached
	set attachedFr $w.mframe
	init_frame $detachedFr.f $det_msg
	init_frame $attachedFr.f $att_msg

# AS: end initialization of attachment

	switchvisual 0
	$this-c startup
    }
    method bindEvents {w} {
	bind $w <Expose> "$this-c redraw"
	bind $w <Configure> "$this-c redraw"

	bind $w <ButtonPress-1> "$this-c mtranslate start %x %y"
	bind $w <Button1-Motion> "$this-c mtranslate move %x %y"
	bind $w <ButtonRelease-1> "$this-c mtranslate end %x %y"
	bind $w <ButtonPress-2> "$this-c mrotate start %x %y %t"
	bind $w <Button2-Motion> "$this-c mrotate move %x %y %t"
	bind $w <ButtonRelease-2> "$this-c mrotate end %x %y %t"
	bind $w <ButtonPress-3> "$this-c mscale start %x %y"
	bind $w <Button3-Motion> "$this-c mscale move %x %y"
	bind $w <ButtonRelease-3> "$this-c mscale end %x %y"

	bind $w <Control-ButtonPress-1> "$this-c mdolly start %x %y"
	bind $w <Control-Button1-Motion> "$this-c mdolly move %x %y"
	bind $w <Control-ButtonRelease-1> "$this-c mdolly end %x %y"
	bind $w <Control-ButtonPress-2> "$this-c mrotate_eyep start %x %y %t"
	bind $w <Control-Button2-Motion> "$this-c mrotate_eyep move %x %y %t"
	bind $w <Control-ButtonRelease-2> "$this-c mrotate_eyep end %x %y %t"
	bind $w <Control-ButtonPress-3> "$this-c municam start %x %y %t"
	bind $w <Control-Button3-Motion> "$this-c municam move %x %y %t"
	bind $w <Control-ButtonRelease-3> "$this-c municam end %x %y %t"

	bind $w <Shift-ButtonPress-1> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-ButtonPress-2> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-ButtonPress-3> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-Button1-Motion> "$this-c mpick move %x %y %s 1"
	bind $w <Shift-Button2-Motion> "$this-c mpick move %x %y %s 2"
	bind $w <Shift-Button3-Motion> "$this-c mpick move %x %y %s 3"
	bind $w <Shift-ButtonRelease-1> "$this-c mpick end %x %y %s %b"
	bind $w <Shift-ButtonRelease-2> "$this-c mpick end %x %y %s %b"
	bind $w <Shift-ButtonRelease-3> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonPress-1> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-ButtonPress-2> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-ButtonPress-3> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-Button1-Motion> "$this-c mpick move %x %y %s 1"
	bind $w <Lock-Button2-Motion> "$this-c mpick move %x %y %s 2"
	bind $w <Lock-Button3-Motion> "$this-c mpick move %x %y %s 3"
	bind $w <Lock-ButtonRelease-1> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonRelease-2> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonRelease-3> "$this-c mpick end %x %y %s %b"
    }

    method killWindow { vw } {
        set w .ui[modname]
	if {"$vw"=="$w"} {
	    $this-c killwindow
	}
    }

    method removeMFrame {w} {

	if { $IsAttached!=0 } {
	    pack forget $attachedFr
	    append geom [winfo width $w] x [expr [winfo height $w]-[winfo height $w.mframe]]
	    wm geometry $w $geom
	    update
	} else { 
	    wm withdraw $detachedFr
	}
	
	$w.bframe.more configure -command "$this addMFrame $w" -text "+"
	set IsDisplayed 0
    }
    
    method addMFrame {w} {

	if { $IsAttached!=0} {
	    pack $attachedFr -anchor w -side top -after $w.bframe
	    append geom [expr [winfo width $w]>[winfo width $w.mframe] ?[winfo width $w]:[winfo width $w.mframe]] x [expr [winfo height $w]+[winfo reqheight $w.mframe]]
	    wm geometry $w $geom
	    update
	} else {
	    wm deiconify $detachedFr
	}
	$w.bframe.more configure -command "$this removeMFrame $w" -text "-"
	set IsDisplayed 1
    }

    method init_frame {m msg} {
	if { [winfo exists $m] } {
	global "$this-global-light"
	global "$this-global-fog"
	global "$this-global-type"
	global "$this-global-debug"
	global "$this-global-clip"
	global "$this-global-cull"
	global "$this-global-dl"
	global "$this-global-movie"
	global "$this-global-movieName"
	global "$this-global-movieFrame"
	global "$this-global-resize"
	global "$this-x-resize"
	global "$this-y-resize"
	global $this-do_stereo
	global $this-sbase
	global $this-sr
	global $this-do_bawgl
	global $this-tracker_state
	
	set "$this-global-light" 1
	set "$this-global-fog" 0
	set "$this-global-type" Gouraud
	set "$this-global-debug" 0
	set "$this-global-clip" 1
	set "$this-global-cull" 0
	set "$this-global-dl" 0
	set "$this-global-movie" 0
	set "$this-global-movieName" "movie"
	set "$this-global-movieFrame" 0
	set "$this-global-resize" 0
	set "$this-x-resize" 700
	set "$this-y-resize" 512
	set $this-do_bawgl 0
	set $this-tracker_state 0
	
	set r "$this-c redraw"
	bind $m <Double-ButtonPress-1> "$this switch_frames"

	label $m.cut -anchor w -text $msg -font "-Adobe-Helvetica-bold-R-Normal-*-12-75-*"
	pack $m.cut -side top -anchor w -pady 5 -padx 5
	bind $m.cut <Double-ButtonPress-1> "$this switch_frames"
	
	frame $m.eframe
	
	checkbutton $m.eframe.light -text Lighting -variable $this-global-light \
	    -command "$this-c redraw"
	checkbutton $m.eframe.fog -text Fog -variable $this-global-fog \
	    -command "$this-c redraw"
	checkbutton $m.eframe.bbox -text BBox -variable $this-global-debug \
	    -command "$this-c redraw"
	checkbutton $m.eframe.clip -text "Use Clip" -variable $this-global-clip \
	    -command "$this-c redraw"
	checkbutton $m.eframe.cull -text "Back Cull" -variable $this-global-cull \
	    -command "$this-c redraw"
	checkbutton $m.eframe.dl -text "Display List" \
	    -variable $this-global-dl -command "$this-c redraw"
	
# 	checkbutton $m.eframe.movie -text "Save Movie" -variable $this-global-movie
# 	frame $m.eframe.mf
# 	label $m.eframe.mf.lf -text "  Frame: "
# 	entry $m.eframe.mf.vf -relief sunken -width 4 -textvariable $this-global-movieFrame
# 	pack $m.eframe.mf.lf $m.eframe.mf.vf -side left
	
# 	frame $m.eframe.mn
# 	label $m.eframe.mn.ln -text "  Name: "
# 	entry $m.eframe.mn.vn -relief sunken -width 4 -textvariable $this-global-movieName
# 	pack $m.eframe.mn.ln $m.eframe.mn.vn -side left
	
# 	pack $m.eframe -anchor w -padx 2 -side left
# 	pack  $m.eframe.light $m.eframe.fog $m.eframe.bbox $m.eframe.clip \
# 		$m.eframe.cull $m.eframe.dl $m.eframe.movie $m.eframe.mf \
#	         $m.eframe.mn -in $m.eframe -side top -anchor w
	
	pack $m.eframe -anchor w -padx 2 -side left
	pack  $m.eframe.light $m.eframe.fog $m.eframe.bbox $m.eframe.clip \
		$m.eframe.cull $m.eframe.dl -in $m.eframe -side top -anchor w

        frame $m.eframe.f -relief groove -borderwidth 2
        pack $m.eframe.f -side top -anchor w
        label $m.eframe.f.l -text "Record Movie as:"
        pack $m.eframe.f.l -side top 
	checkbutton $m.eframe.f.resize -text "Resize: " \
	    -variable $this-global-resize \
	    -offvalue 0 -onvalue 1 -command "$this resize; $this-c redraw"
	entry $m.eframe.f.e1 -textvariable $this-x-resize -width 4
	label $m.eframe.f.x -text x
	entry $m.eframe.f.e2 -textvariable $this-y-resize -width 4
#         checkbutton $m.eframe.f.resize -text "Resize 352x240" \
# 	    -variable $this-global-resize \
# 	    -offvalue 0 -onvalue 1 -command "$this resize; $this-c redraw"
#         checkbutton $m.eframe.f.resize2 -text "Resize 1024x768" \
# 	    -variable $this-global-resize \
# 	    -offvalue 0 -onvalue 2 -command "$this resize; $this-c redraw"
#         checkbutton $m.eframe.f.resize3 -text "Resize 1600x1024" \
# 	    -variable $this-global-resize \
# 	    -offvalue 0 -onvalue 3 -command "$this resize; $this-c redraw"
        radiobutton $m.eframe.f.none -text "Stop Recording" \
            -variable $this-global-movie -value 0 -command "$this-c redraw"
	radiobutton $m.eframe.f.raw -text "Raw Frames" \
            -variable $this-global-movie -value 1 -command "$this-c redraw"
	if { [$this-c have_mpeg] } {
	    radiobutton $m.eframe.f.mpeg -text "Mpeg" -variable \
		    $this-global-movie -value 2 -command "$this-c redraw"
	} else {
	    radiobutton $m.eframe.f.mpeg -text "Mpeg" \
		    -variable $this-global-movie -value 2 \
		    -state disabled -disabledforeground "" \
		    -command "$this-c redraw"
	}
        entry $m.eframe.f.moviebase -relief sunken -width 12 \
	    -textvariable "$this-global-movieName" 
        pack $m.eframe.f.none $m.eframe.f.raw $m.eframe.f.mpeg \
            -side top  -anchor w
        pack $m.eframe.f.moviebase -side top -anchor w -padx 2 -pady 2
	pack $m.eframe.f.resize $m.eframe.f.e1 \
	    $m.eframe.f.x $m.eframe.f.e2 -side left  -anchor w

	make_labeled_radio $m.shade "Shading:" $r top $this-global-type \
		{Wire Flat Gouraud}
	pack $m.shade -in $m.eframe -side top -anchor w

	frame $m.objlist -relief groove -borderwidth 2
	pack $m.objlist -side left -padx 2 -pady 2 -fill y
	label $m.objlist.title -text "Objects:"
	pack $m.objlist.title -side top
	canvas $m.objlist.canvas -width 370 -height 100 \
	        -scrollregion "0 0 370 100" \
		-yscrollcommand "$m.objlist.scroll set" -borderwidth 0 -yscrollincrement 10
	pack $m.objlist.canvas -side right -padx 2 -pady 2 -fill y
	
	frame $m.objlist.canvas.frame -relief sunken -borderwidth 2
	pack $m.objlist.canvas.frame
	$m.objlist.canvas create window 0 1 -window $m.objlist.canvas.frame \
		-anchor nw
	
	scrollbar $m.objlist.scroll -relief sunken \
		-command "$m.objlist.canvas yview"
	pack $m.objlist.scroll -fill y -side right -padx 2 -pady 2
	
        # CollabVis code begin
        if {[set $this-have_collab_vis]} {
	    checkbutton $m.view_server -text "Remote" -variable \
                $this-view_server -onvalue 2 -offvalue 0 \
                -command "$this-c doServer"
	    pack $m.view_server -side top
        }
	# CollabVis code end

        checkbutton $m.caxes -text "Show Axes" -variable $this-caxes -onvalue 1 -offvalue 0 -command "$this-c centerGenAxes; $this-c redraw"
        checkbutton $m.raxes -text "Orientation" -variable $this-raxes -onvalue 1 -offvalue 0 -command "$this-c rotateGenAxes; $this-c redraw"
	checkbutton $m.ortho -text "Ortho View" -variable $this-ortho-view -onvalue 1 -offvalue 0 -command "$this-c redraw"
	# checkbutton $m.iaxes -text "Icon Axes" -variable $this-iaxes -onvalue 1 -offvalue 0 -command "$this-c iconGenAxes; $this-c redraw"
	# pack $m.caxes $m.iaxes -side top
	pack $m.caxes -side top -anchor w
	pack $m.raxes -side top -anchor w
	pack $m.ortho -side top -anchor w
    
	checkbutton $m.stereo -text "Stereo" -variable $this-do_stereo \
		-command "$this-c redraw"
	pack $m.stereo -side top -anchor w
	
	scale $m.sbase -variable $this-sbase -length 100 -from 0.1 -to 2 \
		-resolution 0.02 -orient horizontal -label "Fusion Scale:" \
		-command "$this-c redraw"
	pack $m.sbase -side top -anchor w
#	checkbutton $m.sr -text "Fixed\nFocal\nDepth" -variable $this-sr -anchor w
#	pack $m.sr -side top
	
	# the stuff below doesn't have corresponding c-functions
	
#	checkbutton $m.tracker -text "Tracker" -variable $this-tracker_state \
#		-command "$this-c tracker"
#	pack $m.tracker -side top

	
#	checkbutton $m.bench -text "SCIBench" -variable $this-do_bawgl \
#                -command "$this bench $this-do_bawgl"
#        pack $m.bench -side top

#	button $m.tracker_reset -text " Reset\nTracker" \
#		-command "$this-c reset_tracker"
#	pack $m.tracker_reset -side top
#        } else {
#	    puts "Non-existing frame to initialize!"
#	}

	bind $m.eframe.f.e1 <Return> "$this resize"
	bind $m.eframe.f.e2 <Return> "$this resize"
	if {[set $this-global-resize] == 0} {
	    set color "#505050"
	    $m.eframe.f.x configure -foreground $color
	    $m.eframe.f.e1 configure -state disabled -foreground $color
	    $m.eframe.f.e2 configure -state disabled -foreground $color
	}
    }

    method resize { } {
	set w .ui[modname]
	if { [set $this-global-resize] == 0 } {
	    wm geometry $w "="
	    pack configure $w.wframe -expand yes -fill both

	    set color "#505050"
	    if { $IsAttached == 1 } {
		set m $w.mframe.f
		$m.eframe.f.x configure -foreground $color
		$m.eframe.f.e1 configure -state disabled -foreground $color
		$m.eframe.f.e2 configure -state disabled -foreground $color
	    } else {
		set m $w.detached.f
		$m.eframe.f.x configure -foreground $color
		$m.eframe.f.e1 configure -state disabled -foreground $color
		$m.eframe.f.e2 configure -state disabled -foreground $color
	    }
	} else {
	    if { $IsAttached == 1 } { $this switch_frames }
	    set m $w.detached.f
	    set xsize [set $this-x-resize]
	    set ysize [set $this-y-resize]
	    set size "$xsize\x$ysize"
	    set xsize [expr $xsize + 14]
	    set ysize [expr $ysize + 123]
	    set geomsize "$xsize\x$ysize"
	    wm geometry $w "=$geomsize"
	    pack configure $w.wframe -expand no -fill none
	    $w.wframe.draw configure -geometry $size
	    $m.eframe.f.x configure -foreground black
	    $m.eframe.f.e1 configure -state normal -foreground black
	    $m.eframe.f.e2 configure -state normal -foreground black
	}
    }

    method switch_frames {} {
	set w .ui[modname]
	if {$IsDisplayed!=0} {
#	    update
# getting current window position
#	    set geom [wm geometry $w]
#	    set f [string first "+" $geom]
#	    set s [string first "-" $geom]
#	    if { [expr $f >= 0 && $s>=0] } {    
#		set ind [expr $f>$s ? $s:$f]
#	    } else {
#		set ind [expr $f>$s ? $f:$s]
#	    }
		
#	    if {$ind >=0} {
#		set pos [string range $geom $ind [expr [string length $geom]-1]]
#	    } else {
#		set pos ""
#	    }

#	    set geom ""

	    # handling main window resizing by hand
  	    
	    if { $IsAttached!=0} {
		pack forget $attachedFr
		
		append geom [winfo width $w] x [expr [winfo height $w]-[winfo reqheight $w.mframe]]
		wm geometry $w $geom
		wm deiconify $detachedFr
		set IsAttached 0
	    } else {
		wm withdraw $detachedFr
		
		pack $attachedFr -anchor w -side top -after $w.bframe
		append geom [winfo width $w] x [expr [winfo height $w]+[winfo reqheight $w.mframe]]
		wm geometry $w $geom
		set IsAttached 1
	    }
	    update
	}
    }

    method updatePerf {p1 p2 p3} {
	set w .ui[modname]
	$w.bframe.pf.perf1 configure -text $p1
	$w.bframe.pf.perf2 configure -text $p2
	$w.bframe.pf.perf3 configure -text $p3
    }

    method switchvisual {idx} {
	set w .ui[modname]
	if {[winfo exists $w.wframe.draw]} {
	    destroy $w.wframe.draw
	}
	$this-c switchvisual $w.wframe.draw $idx 640 512
	if {[winfo exists $w.wframe.draw]} {
	    bindEvents $w.wframe.draw
	    pack $w.wframe.draw -expand yes -fill both
	}
    }	

    method bench {bench} {
        upvar #0 $bench b
        set w .ui[modname]
        puts $w
        if {$b == 1} {
            if {[winfo exists $w.wframe.draw]} {
                destroy $w.wframe.draw
		destroy $w.wframe
            }
            toplevel $w.wframe -borderwidth 1
            wm overrideredirect $w.wframe 1
            wm geometry $w.wframe 1024x768+1280+0
            $this-c switchvisual $w.wframe.draw 0 1024 768
            if {[winfo exists $w.wframe.draw]} {
                bind $w <KeyPress-Escape> "$w.mframe.f.bench invoke"
		pack $w.wframe.draw -expand yes -fill both
		$this-c startbawgl
	    }
        } else {
            if {[winfo exists $w.wframe.bench.draw]} {
                $this-c stopbawgl
		bind $w <KeyPress-Escape> ""
		destroy $w.wframe.bench.draw
            }
            destroy $w.wframe
	    frame $w.wframe
            pack $w.wframe
            $this-c switchvisual $w.wframe.draw 0 640 512
            if {[winfo exists $w.wframe.draw]} {
                bindEvents $w.wframe.draw
                pack $w.wframe.draw -expand yes -fill both
            }
        }
    }

    method makeViewPopup {} {
	set w .view[modname]
	toplevel $w
	wm title $w "View"
	wm iconname $w view
	wm minsize $w 100 100
	set c "$this-c redraw "
	set view $this-view
	makePoint $w.eyep "Eye Point" $view-eyep $c
	pack $w.eyep -side left -expand yes -fill x
	makePoint $w.lookat "Look at Point" $view-lookat $c
	pack $w.lookat -side left -expand yes -fill x
	makeNormalVector $w.up "Up Vector" $view-up $c
	pack $w.up -side left -expand yes -fill x
	global $view-fov
	frame $w.f -relief groove -borderwidth 2
	pack $w.f
	scale $w.f.fov -orient horizontal -variable $view-fov \
		-from 0 -to 180 -label "Field of View:" \
		-showvalue true -tickinterval 90 \
		-digits 3 \
		-command $c
	pack $w.f.fov -expand yes -fill x
#  	entry $w.f.fove -textvariable $view-fov
#  	pack $w.f.fove -side top -expand yes -fill x
#  	bind $w.f.fove <Return> "$command $view-fov"
    }

    method makeSceneMaterialsPopup {} {
	set w .scenematerials[modname]
	toplevel $w
	wm title $w "Scene Materials"
	wm iconname $w materials
	wm minsize $w 100 100
	set c "$this-c redraw "

	frame $w.ambient
	label $w.ambient.l -width 16 -text "Ambient Scale"
	global $this-ambient-scale
	entry $w.ambient.e -relief sunken -width 6 \
		-textvariable $this-ambient-scale
	bind $w.ambient.e <Return> $c
	pack $w.ambient.l $w.ambient.e -side left -fill x

	frame $w.diffuse
	label $w.diffuse.l -width 16 -text "Diffuse Scale"
	global $this-diffuse-scale
	entry $w.diffuse.e -relief sunken -width 6 \
		-textvariable $this-diffuse-scale
	bind $w.diffuse.e <Return> $c
	pack $w.diffuse.l $w.diffuse.e -side left -fill x

	frame $w.specular
	label $w.specular.l -width 16 -text "Specular Scale"
	global $this-specular-scale
	entry $w.specular.e -relief sunken -width 6 \
		-textvariable $this-specular-scale
	bind $w.specular.e <Return> $c
	pack $w.specular.l $w.specular.e -side left -fill x

	frame $w.shininess
	label $w.shininess.l -width 16 -text "Shininess Scale"
	global $this-shininess-scale
	entry $w.shininess.e -relief sunken -width 6 \
		-textvariable $this-shininess-scale
	bind $w.shininess.e <Return> $c
	pack $w.shininess.l $w.shininess.e -side left -fill x

	frame $w.emission
	label $w.emission.l -width 16 -text "Emission Scale"
	global $this-emission-scale
	entry $w.emission.e -relief sunken -width 6 \
		-textvariable $this-emission-scale
	bind $w.emission.e <Return> $c
	pack $w.emission.l $w.emission.e -side left -fill x

	pack $w.ambient $w.diffuse $w.specular $w.emission $w.shininess \
		-side top -fill both
    }

    method makeBackgroundPopup {} {
	set w .bg[modname]
	toplevel $w
	wm title $w "Background"
	wm iconname $w background
	wm minsize $w 100 100
	set c "$this-c redraw "
	makeColorPicker $w $this-bgcolor $c ""
    }

    method updateMode {msg} {
	global .ui[modname].bframe.mousemode
	set mouseModeText .ui[modname].bframe.mousemode
	$mouseModeText itemconfigure mouseModeText -text $msg
    }   

    method addObject {objid name} {
	addObjectToFrame $objid $name $detachedFr
	addObjectToFrame $objid $name $attachedFr
    }

    method addObjectToFrame {objid name frame} {
	set w .ui[modname]
	set m $frame.f
	frame  $m.objlist.canvas.frame.objt$objid
	checkbutton $m.objlist.canvas.frame.obj$objid -text $name \
		-relief flat -variable "$this-$name" -command "$this-c redraw"
	
	set newframeheight [winfo reqheight $m.objlist.canvas.frame.obj$objid]
	
	set menun $m.objlist.canvas.frame.menu$objid.menu

	menubutton $m.objlist.canvas.frame.menu$objid -text Options \
		-relief raised -menu $menun
	menu $menun
	$menun add checkbutton -label Lighting -variable $this-$objid-light \
		-command "$this-c redraw"
	$menun add checkbutton -label BBox -variable $this-$objid-debug \
		-command "$this-c redraw"
	$menun add checkbutton -label Fog -variable $this-$objid-fog \
		-command "$this-c redraw"
	$menun add checkbutton -label "Use Clip" -variable $this-$objid-clip \
		-command "$this-c redraw"
	$menun add checkbutton -label "Back Cull" -variable $this-$objid-cull \
		-command "$this-c redraw"
	$menun add checkbutton -label "Display List" -variable $this-$objid-dl\
		-command "$this-c redraw"
	$menun add separator
	$menun add radiobutton -label Wire -variable $this-$objid-type
	$menun add radiobutton -label Flat -variable $this-$objid-type
	$menun add radiobutton -label Gouraud -variable $this-$objid-type
	global "$this-$objid-light"
	global "$this-$objid-fog"
	global "$this-$objid-type"
	global "$this-$objid-debug"
	global "$this-$objid-clip"
	global "$this-$objid-cull"
	global "$this-$objid-dl"

	set "$this-$objid-type" Gouraud
	set "$this-$objid-light" 1
	set "$this-$objid-fog" 0
	set "$this-$objid-debug" 0
	set "$this-$objid-clip" 1
	set "$this-$objid-cull" 0
	set "$this-$objid-dl" 0


	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
	#tkwait visibility $m.objlist.canvas.frame.obj$objid
	update idletasks
	set width [winfo width $m.objlist.canvas.frame]
	#set height [winfo height $m.objlist.canvas.frame]
	set height [lindex [$m.objlist.canvas cget -scrollregion] end]

	incr height [expr $newframeheight+20]

	$m.objlist.canvas configure -scrollregion "0 0 $width $height"

	set view [$m.objlist.canvas yview]
	$m.objlist.scroll set [lindex $view 0] [lindex $view 1]
    }

    method addObject2 {objid} {
	addObjectToFrame_2 $objid $detachedFr
	addObjectToFrame_2 $objid $attachedFr
    }
    
    method addObjectToFrame_2 {objid frame} {
	set w .ui[modname]
	set m $frame.f
	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
    }
    

    method removeObject {objid} {
	removeObjectFromFrame $objid $detachedFr
	removeObjectFromFrame $objid $attachedFr
    }

    method removeObjectFromFrame {objid frame} {
	set w .ui[modname]
	set m $frame.f
	pack forget $m.objlist.canvas.frame.objt$objid
    }

    method makeLineWidthPopup {} {
	set w .lineWidth[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm title $w "Line Width"
	wm minsize $w 250 100
	frame $w.f
	global $this-line-width
	scale $w.f.scale -command "$this-c redraw" -variable \
		$this-line-width -orient horizontal -from 1 -to 5 \
		-resolution .1 -showvalue true -tickinterval 1 -digits 0 \
		-label "Line Width:"
	pack $w.f.scale -fill x -expand 1
	pack $w.f -fill x -expand 1
    }	

    method makePolygonOffsetPopup {} {
	set w .polygonOffset[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm title $w "Polygon Offset"
	wm minsize $w 250 100
	frame $w.f
	global $this-polygon-offset-factor
	global $this-polygon-offset-units
	scale $w.f.factor -command "$this-c redraw" -variable \
		$this-polygon-offset-factor -orient horizontal -from -4 \
		-to 4 -resolution .01 -showvalue true -tickinterval 2 \
		-digits 3 -label "Offset Factor:"
	scale $w.f.units -command "$this-c redraw" -variable \
		$this-polygon-offset-units -orient horizontal -from -4 \
		-to 4 -resolution .01 -showvalue true -tickinterval 2 \
		-digits 3 -label "Offset Units:"
#	pack $w.f.factor $w.f.units -fill x -expand 1 -pady 10
	pack $w.f.factor -fill x -expand 1
	pack $w.f -fill x -expand 1
    }	

    method makePointSizePopup {} {
	set w .psize[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm title $w "Point Size"
	wm minsize $w 250 100 

	frame $w.f
	global $this-point-size
	scale $w.f.scale -command "$this-c redraw" -variable \
		$this-point-size -orient horizontal -from 1 -to 5 \
		-resolution .1 -showvalue true -tickinterval 1 -digits 0 \
		-label "Pixel Size:"
	pack $w.f.scale -fill x -expand 1
	pack $w.f -fill x -expand 1
    }	

    method makeClipPopup {} {
	set w .clip[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm title $w "Clipping Planes"
	wm minsize $w 200 100 
	set clip $this-clip

	global $clip-num
	set $clip-num 6

	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-visible
	set $clip-visible 0
	set $clip-normal-d 0.0
	set $clip-normal-x 1.0
	set $clip-normal-y 0.0
	set $clip-normal-z 0.0

	for {set i 1} {$i <= [set $clip-num]} {incr i 1} {
	    set mod $i


	    global $clip-normal-x-$mod
	    global $clip-normal-y-$mod
	    global $clip-normal-z-$mod
	    global $clip-normal-d-$mod
	    global $clip-visible-$mod
	    set $clip-visible-$mod 0
	    set $clip-normal-d-$mod 0.0
	    set $clip-normal-x-$mod 1.0
	    set $clip-normal-y-$mod 0.0
	    set $clip-normal-z-$mod 0.0
	}
	set c "$this setClip ; $this-c redraw"
	global $clip-selected
	set $clip-selected 1
	set menup [tk_optionMenu $w.which $clip-selected 1 2 3 4 5 6]

	for {set i 0}  {$i < [set $clip-num]} {incr i 1} {
	    $menup entryconfigure $i -command "[$menup entrycget $i -command] ; $this useClip"
	}
	
	pack $w.which
	checkbutton $w.visibile -text "Visible" -relief flat \
		-variable "$clip-visible" -command "$this setClip ; $this-c redraw"
	pack $w.visibile

	makePlane $w.normal "Plane Normal" $clip-normal $c
	pack $w.normal -side left -expand yes -fill x
	frame $w.f -relief groove -borderwidth 2
	pack $w.f -expand yes -fill x
    }

    method useClip {} {
	set clip $this-clip
	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-visible
	global $clip-selected
	set cs [set $clip-selected]

	global $clip-normal-x-$cs
	global $clip-normal-y-$cs
	global $clip-normal-z-$cs
	global $clip-normal-d-$cs
	global $clip-visible-$cs

	set $clip-normal-x [set $clip-normal-x-$cs]
	set $clip-normal-y [set $clip-normal-y-$cs]
	set $clip-normal-z [set $clip-normal-z-$cs]
	.clip[modname].normal.e newvalue [set $clip-normal-d-$cs]
	set $clip-visible [set $clip-visible-$cs]
    }

    method setClip {} {
	set clip $this-clip
	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-visible
	global $clip-selected
	set cs [set $clip-selected]

	global $clip-normal-x-$cs
	global $clip-normal-y-$cs
	global $clip-normal-z-$cs
	global $clip-normal-d-$cs
	global $clip-visible-$cs

	#set n $clip-normal-x-$cs
	#puts "set $n [set $clip-normal-x]"
	set  $clip-normal-x-$cs [set $clip-normal-x]
	set  $clip-normal-y-$cs [set $clip-normal-y]
	set  $clip-normal-z-$cs [set $clip-normal-z]
	set  $clip-normal-d-$cs [set $clip-normal-d]
	set  $clip-visible-$cs [set $clip-visible]
    }

    method invertClip {} {
	set clip $this-clip
	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-selected
	set cs [set $clip-selected]

	global $clip-normal-x-$cs
	global $clip-normal-y-$cs
	global $clip-normal-z-$cs
	
	set  $clip-normal-x-$cs [expr -1 * [set $clip-normal-x]]
	set  $clip-normal-y-$cs [expr -1 * [set $clip-normal-y]]
	set  $clip-normal-z-$cs [expr -1 * [set $clip-normal-z]]

	set $clip-normal-x [set $clip-normal-x-$cs]
	set $clip-normal-y [set $clip-normal-y-$cs]
	set $clip-normal-z [set $clip-normal-z-$cs]
    }

    method makeAnimationPopup {} {
	set w .anim[modname]
	toplevel $w
	wm title $w "Animation"
	wm iconname $w "Animation"
	wm minsize $w 100 100
	frame $w.ctl
	pack $w.ctl -side top -fill x
	set afont "-adobe-helvetica-bold-r-*-*-24-*-*-*-*-*-*-*"
	button $w.ctl.rstep -text "<-" -font $afont \
		-command "$this rstep"
	button $w.ctl.rew -text "<<" -font $afont \
		-command "$this rew"
	button $w.ctl.rplay -text "<" -font $afont \
		-command "$this rplay"
	button $w.ctl.stop -text "\[\]" -font $afont \
		-command "$this stop"
	button $w.ctl.play -text ">" -font $afont \
		-command "$this play"
	button $w.ctl.ff -text ">>" -font $afont \
		-command "$this ff"
	button $w.ctl.step -text "->" -font $afont \
		-command "$this step"
	pack $w.ctl.rstep $w.ctl.rew $w.ctl.rplay $w.ctl.stop \
		$w.ctl.play $w.ctl.ff $w.ctl.step \
		-side left -ipadx 3 -ipady 3

	scale $w.rate -orient horizontal -variable $this-framerate \
		-from 0 -to 60 -label "Frame rate:" \
		-showvalue true -tickinterval 10
	pack $w.rate -side top -fill x
	frame $w.arate
	pack $w.arate -side top -fill x
	label $w.arate.lab -text "Actual Rate:"
	pack $w.arate.lab -side left
	label $w.arate.value -text ""
	pack $w.arate.value -side left

	scale $w.tframes -orient horizontal \
		-from 0 -to 300 -label "Total frames:" \
		-variable $this-totframes \
		-showvalue true -tickinterval 10
	pack $w.tframes -side top -fill x
	scale $w.tbeg -orient horizontal -variable $this-tbeg \
		-from 0 -to 1 -label "Begin time:" \
		-resolution 0.001 -digits 4 \
		-showvalue true -tickinterval 2
	scale $w.tend -orient horizontal -variable $this-tend \
		-from 0 -to 1 -label "End time:" \
		-resolution 0.001 -digits 4 \
		-showvalue true -tickinterval 2
	scale $w.ct -orient horizontal -variable $this-current_time \
		-from 0 -to 1 -label "Current time:" \
		-resolution 0.001 -digits 4 \
		-showvalue true -tickinterval 2 \
		-command "$this-c redraw"
	pack $w.tbeg $w.tend $w.ct -side top -fill x
	entry $w.savefile -textvariable $this-saveprefix
	pack $w.savefile -side top -fill x
    }
    method setFrameRate {rate} {
	set w .anim[modname]
	if {[winfo exists $w]} {
	    $w.arate.value config -text $rate
	    update idletasks
	}
    }
    method frametime {} {
	global $this-tbeg $this-tend $this-totframes
	set tbeg [set $this-tbeg]
	set tend [set $this-tend]
	set tframes [set $this-totframes]
	return [expr ($tend-$tbeg)/$tframes]
    }
    method rstep {} {
	global $this-current_time $this-tbeg
	set frametime [$this frametime]
	set ctime [set $this-current_time]
	set newtime [expr $ctime-$frametime]
	set tbeg [set $this-tbeg]
	if {$newtime < $tbeg} {
	    set newtime $tbeg
	}
	$this-c anim_redraw $newtime $newtime 1 0
    }
    method rew {} {
	global $this-tbeg
	set newtime [set $this-tbeg]
	$this-c anim_redraw $newtime $newtime 1 0
    }
    method rplay {} {
	global $this-current_time $this-tbeg $this-tend \
		$this-framerate $this-totframes
	set ctime [set $this-current_time]
	set tbeg [set $this-tbeg]
	set tend [set $this-tend]
	set frametime [$this frametime]
	if {$ctime < [expr $tbeg+$frametime]} {
	    set ctime $tend
	}
	set framerate [set $this-framerate]
	set totframes [set $this-totframes]
	set nframes [expr ($ctime-$tbeg)/($tend-$tbeg)*$totframes]
	$this-c anim_redraw $ctime $tbeg $nframes $framerate
    }

    method play {} {
	global $this-current_time $this-tbeg $this-tend \
		$this-framerate $this-totframes
	set ctime [set $this-current_time]
	set tbeg [set $this-tbeg]
	set tend [set $this-tend]
	set frametime [$this frametime]
	if {$ctime > [expr $tend-$frametime]} {
	    set ctime $tbeg
	}
	set framerate [set $this-framerate]
	set totframes [set $this-totframes]
	set nframes [expr ($tend-$ctime)/($tend-$tbeg)*$totframes]
	$this-c anim_redraw $ctime $tend $nframes $framerate
    }
    method step {} {
	global $this-current_time $this-tend
	set frametime [$this frametime]
	set ctime [set $this-current_time]
	set newtime [expr $ctime+$frametime]
	set tend [set $this-tend]
	if {$newtime > $tend} {
	    set newtime $tend
	}
	$this-c anim_redraw $newtime $newtime 1 0
    }
    method ff {} {
	global $this-tend
	set newtime [set $this-tend]
	$this-c anim_redraw $newtime $newtime 1 0
    }
    method crap {} {
	make_labeled_radio $w.sw "Animation:" "$this-c redraw" \
		left $this-do_animation \
		{ {On 1} {Off 0} }
	scale $w.anim.tbeg -orient horizontal -variable $this-tbeg \
		-from 0 -to 30 -label "Begin Time:" \
		-showvalue true -tickinterval 10
	scale $w.anim.tend -orient horizontal -variable $this-tend \
		-from 0 -to 30 -label "End Time:" \
		-showvalue true -tickinterval 10
	scale $w.anim.nsteps -orient horizontal -variable $this-ntimesteps \
		-from 0 -to 100 -label "Timesteps:" \
		-showvalue true -tickinterval 20
	scale $w.anim.atime -orient horizontal -variable $this-animation_time \
		-from 0 -to 10 -label "Time:" \
		-showvalue true -tickinterval 2
	button $w.anim.go -text "Go" -command "$this-c redraw"
    }
    method translate {axis amt} {
	puts "translate $axis by $amt"
    }
    method rotate {axis amt} {
	puts "rotate $axis by $amt"
    }
    method rscale {amt} {
	puts "scale by $amt"
    }
    method zoom {amt} {
	puts "zoom by $amt"
    }
    method pan {amt} {
	puts "pan by $amt"
    }
    method tilt {amt} {
	puts "tilt by $amt"
    }
    method fov {amt} {
	puts "fov by $amt"
    }

    method makeSaveObjectsPopup {} {
	toplevel .ui[modname]-save
	global $this-saveobjfile $this-saveformat
	set $this-saveobjfile "out.geom"
	makeFilebox .ui[modname]-save $this-saveobjfile \
		"$this doSaveObjects" "destroy .ui[modname]-save"
	set ex .ui[modname]-save.f.extra
	radiobutton $ex.geomb -variable $this-saveformat \
		-text "Dataflow geom object file (Binary)" -value "scirun_binary"
	radiobutton $ex.geoma -variable $this-saveformat \
		-text "Dataflow geom object file (ASCII)" -value "scirun_ascii"
	radiobutton $ex.vrml -variable $this-saveformat \
		-text "VRML file" -value "vrml"
	radiobutton $ex.rib -variable $this-saveformat \
		-text "RenderMan RIB file" -value "rib"
	$ex.geomb select
	pack $ex.geomb $ex.geoma $ex.vrml $ex.rib -side top -anchor w
    }
    method doSaveObjects {} {
	global $this-saveobjfile $this-saveformat
	$this-c saveobj [set $this-saveobjfile] [set $this-saveformat]
    }

    method do_validate_x {path} {
	global $this-resx
	global $this-resy
	global $this-aspect
	global $this_aspect-rat
	puts "do_validate_x: "
	puts [set $this-resx]
	puts [set $this-resy]
	if {[set $this-aspect]} {
	    puts "changing y"
	    set $this-resy [expr [set $this-resx] * 
	                    [expr 1 / [set $this-aspect-rat]]]
	    
	}
	puts $path
	return 1
    }

    method do_validate_y {path} {
	global $this-resx
	global $this-resy
	puts "do_validate_y: "
	puts [set $this-resx]
	puts [set $this-resy]
	if {[string length [set $this-resy]] > 3} { 
	    set $this-resy "0"
	    return 0 
	}
	puts $path
	return 1
    }

    method do_aspect {t} {
	puts "do_aspect: "
	puts $t
    }

    method makeLightSources {} {
	set $this-resx [winfo width .ui[modname].wframe.draw]
	set $this-resy [winfo height .ui[modname].wframe.draw]
	
	toplevel .ui[modname]-lightSources
	set w .ui[modname]-lightSources
	frame $w.tf -relief flat
	pack $w.tf -side top
	frame $w.bf -relief flat
	pack $w.bf -side top
	set i 0
	for { } {$i < 4} {incr i 1} {
	    $this makeLightControl $w.tf $i
	}
# 	for { } {$i < 8} {incr i 1} {
# 	    $this makeLightControl $w.bf $i
# 	}

	label $w.l -text \
	    "Click on number to move light. Note: Headlight will not move."
	label $w.o -text \
	    "Click in circle to change light color/brightness"

 	button $w.breset -text "Reset Lights" -command "$this resetLights $w"
	button $w.bclose -text Close -command "destroy $w"
	pack $w.l $w.o $w.breset $w.bclose -side top -expand yes -fill x
    }
	
    method makeLightControl { w i } {
#	global $this-global-light$i
	global $this-global-lights
	frame $w.f$i -relief flat
	pack $w.f$i -side left
	canvas $w.f$i.c -bg "#BDBDBD" -width 100 -height 100
	pack $w.f$i.c -side top
	set c $w.f$i.c
	checkbutton $w.f$i.b$i -text "on/off" \
	    -variable $this-global-light$i \
	    -command "$this lightSwitch $i"
	pack $w.f$i.b$i

	set ir [expr int([lindex [lindex [set $this-lightColors] $i] 0] * 65535)]
	set ig [expr int([lindex [lindex [set $this-lightColors] $i] 1] * 65535)]
	set ib [expr int([lindex [lindex [set $this-lightColors] $i] 2] * 65535)]
       
	set window .ui[modname]
	set color [format "#%04x%04x%04x" $ir $ig $ib]
	set news [$c create oval 5 5 95 95 -outline "#000000" \
		     -fill $color -tags lc ]
	set t  $i
	if { $t == 0 } { set t "HL" }
	set newt [$c create text 50 50 -fill "#555555" -text $t -tags lname ]
	$c bind lname <B1-Motion> "$this moveLight $c $i %x %y"
	$c bind lc <ButtonPress-1> "$this lightColor $w $c $i"
    }

    method lightColor { w c i } {
	global $this-def-color 
	set color $this-def-color
	global $color-r
	global $color-g
	global $color-b 

 	set $color-r [lindex [lindex [set $this-lightColors] $i] 0]
 	set $color-g [lindex [lindex [set $this-lightColors] $i] 1]
 	set $color-b [lindex [lindex [set $this-lightColors] $i] 2]

	if {[winfo exists $w.color]} { destroy $w.color } 
	toplevel $w.color 
	makeColorPicker $w.color $color \
	    "$this setColor $w.color $c $i $color " \
	    "destroy $w.color"
    }
   method setColor { w c  i color} {
       global $color
       global $color-r
       global $color-g
       global $color-b 

       set lightColor [list [set $color-r] \
			   [set $color-g] [set $color-b]]
       set $this-lightColors \
	   [lreplace [set $this-lightColors] $i $i $lightColor]

       set ir [expr int([set $color-r] * 65535)]
       set ig [expr int([set $color-g] * 65535)]
       set ib [expr int([set $color-b] * 65535)]
       
       set window .ui[modname]
       $c itemconfigure lc -fill [format "#%04x%04x%04x" $ir $ig $ib]
       $this lightSwitch $i
       destroy $w
   }
   method resetLights { w } {
	for { set i 0 } { $i < 4 } { incr i 1 } {
	    if { $i == 0 } {
		set $this-global-light$i 1
		$this lightSwitch $i
	    } else {
		if { $i < 4 } {
		    set c $w.tf.f$i.c
		} else {
		    set c $w.bf.f$i.c
		}
		set $this-global-light$i 0
		set coords [$c coords lname]
		set curX [lindex $coords 0]
		set curY [lindex $coords 1]
		set xn [expr 50 - $curX]
		set yn [expr 50 - $curY]
		$c move lname $xn $yn
		set vec [list 0 0 1 ]
		set $this-lightVectors \
		    [lreplace [set $this-lightVectors] $i $i $vec]
		$c itemconfigure lc -fill \
		    [format "#%04x%04x%04x" 65535 65535 65535 ]
		set lightColor [list 1.0 1.0 1.0]
		set $this-lightColors \
		    [lreplace [set $this-lightColors] $i $i $lightColor]
		$this lightSwitch $i
	    }
	}
    }
    method moveLight { c i x y } {
	if { $i == 0 } return
	set cw [winfo width $c]
	set ch [winfo height $c]
	set selected [$c find withtag current]
	set coords [$c coords current]
	set curX [lindex $coords 0]
	set curY [lindex $coords 1]
	set xn $x
	set yn $y
	set len2 [expr (( $x-50 )*( $x-50 ) + ($y-50) * ($y-50))]
	if { $len2 < 2025 } { 
	    $c move $selected [expr $xn-$curX] [expr $yn-$curY]
	} else { 
	    # keep the text inside the circle
	    set scale [expr 45 / sqrt($len2)]
	    set xn [expr 50 + ($x - 50) * $scale]
	    set yn [expr 50 + ($y - 50) * $scale]
	    $c move $selected [expr $xn-$curX] [expr $yn-$curY]
	}
	# now compute the vector, we know x and y, compute z
	if { $len2 >= 2025 } { 
	    set newz 0 
	} else { set newz [expr sqrt(2025 - $len2)]}
	set newx [expr $xn - 50]
	set newy [expr $yn - 50]
	# normalize the vector
	set len3 [expr sqrt($newx*$newx + $newy*$newy + $newz*$newz)]
	set vec [list [expr $newx/$len3] [expr -$newy/$len3] [expr $newz/$len3]]
	set $this-lightVectors \
	    [lreplace [set $this-lightVectors] $i $i $vec]
	if { [set $this-global-light$i] } {
	    $this lightSwitch $i
	}
    }

    method lightSwitch {i} {
	if { [set $this-global-light$i] == 0 } {
	    $this-c edit_light $i 0 [lindex [set $this-lightVectors] $i] \
		[lindex [set $this-lightColors] $i]
	} else {
	    $this-c edit_light $i 1 [lindex [set $this-lightVectors] $i] \
		[lindex [set $this-lightColors] $i]
	}
    }
	
    method makeSaveImagePopup {} {
	global $this-saveFile
	global $this-saveType
	global $this-resx
	global $this-resy
	global $this-aspect

	set $this-resx [winfo width .ui[modname].wframe.draw]
	set $this-resy [winfo height .ui[modname].wframe.draw]
	
	set w .ui[modname]-saveImage

	if {[winfo exists $w]} {
	   raise $w
           return
        }

	toplevel $w

        wm title $w "Save ViewWindow Image"
    
	makeFilebox $w \
	    $this-saveFile "$this doSaveImage" "destroy $w"
	#$w.f.sel.sel configure -textvariable $saveFile
	set ex $w.f.extra
	radiobutton $ex.raw -variable $this-saveType \
	    -text "Raw File" -value "raw" \
	    -command "$this changeName $w raw"
	
	radiobutton $ex.ppm -variable $this-saveType \
	    -text "PPM File" -value "ppm" \
	    -command "$this changeName $w ppm"

	radiobutton $ex.byextension -variable $this-saveType \
	    -text "By Extension" -value "magick" \
	    -command "$this changeName $w jpeg"

	if { [set $this-saveType] == "ppm" } {
	    $ex.ppm select
	} elseif { [set $this-saveType] == "raw" } {
	    $ex.raw select 
	} else { $ex.byextension select }

	label $ex.resxl  -text "X:" 
	entry $ex.resx -width 5 -text $this-resx 
#	-validate all -validatecommand "$this do_validate_x $ex"

	label $ex.resyl  -text "Y:" 
	entry $ex.resy -width 5 -text $this-resy 
#	-validate all	-validatecommand "$this do_validate_y $ex"
	checkbutton $ex.aspect -text "Preserve Aspect Ratio" \
		-variable $this-aspect -command "$this do_aspect $ex" 
	$ex.aspect select
	pack $ex.raw $ex.ppm $ex.byextension -side top -anchor w
	pack $ex.resxl $ex.resx $ex.resyl $ex.resy -side left -anchor w
    }
    
    method changeName { w type} {
	global $this-saveFile
	set name [split [set $this-saveFile] .]
	set newname [lreplace $name end end $type]
	set $this-saveFile [join $newname .]
    }
    method doSaveImage {} {
	global $this-saveFile
	global $this-saveType
	$this-c dump_viewwindow [set $this-saveFile] [set $this-saveType] [set $this-resx] [set $this-resy]
	$this-c redraw
    }
}


catch {rename EmbeddedViewWindow ""}


itcl_class EmbeddedViewWindow {
    public viewer
    
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }

    method set_defaults {} {

	# set defaults values for parameters that weren't set in a script

        # CollabVis code begin 
        global $this-have_collab_vis 
        # CollabVis code end

	global $this-saveFile
	global $this-saveType
	if {![info exists $this-File]} {set $this-saveFile "out.raw"}
	if {![info exists $this-saveType]} {set $this-saveType "raw"}

	# Animation parameters
	global $this-current_time
	if {![info exists $this-current_time]} {set $this-current_time 0}
	global $this-tbeg
	if {![info exists $this-tbeg]} {set $this-tbeg 0}
	global $this-tend
	if {![info exists $this-tend]} {set $this-tend 1}
	global $this-framerate
	if {![info exists $this-framerate]} {set $this-framerate 15}
	global $this-totframes
	if {![info exists $this-totframes]} {set $this-totframes 30}
	global $this-caxes
        if {![info exists $this-caxes]} {set $this-caxes 0}
	global $this-raxes
        if {![info exists $this-raxes]} {set $this-raxes 1}

	# Need to initialize the background color
	global $this-bgcolor-r
	if {![info exists $this-bgcolor-r]} {set $this-bgcolor-r 0}
	global $this-bgcolor-g
	if {![info exists $this-bgcolor-g]} {set $this-bgcolor-g 0}
	global $this-bgcolor-b
	if {![info exists $this-bgcolor-b]} {set $this-bgcolor-b 0}

	# Need to initialize the scene material scales
	global $this-ambient-scale
	if {![info exists $this-ambient-scale]} {set $this-ambient-scale 1.0}
	global $this-diffuse-scale
	if {![info exists $this-diffuse-scale]} {set $this-diffuse-scale 1.0}
	global $this-specular-scale
	if {![info exists $this-specular-scale]} {set $this-specular-scale 0.4}
	global $this-emission-scale
	if {![info exists $this-emission-scale]} {set $this-emission-scale 1.0}
	global $this-shininess-scale
	if {![info exists $this-shininess-scale]} {set $this-shininess-scale 1.0}
	# Initialize point size, line width, and polygon offset
	global $this-point-size
	if {![info exists $this-point-size]} {set $this-point-size 1.0}
	global $this-line-width
	if {![info exists $this-line-width]} {set $this-line-width 1.0}
	global $this-polygon-offset-factor
 	if {![info exists $this-polygon-offset-factor]} \
	    {set $this-polygon-offset-factor 1.0}
	global $this-polygon-offset-units
	if {![info exists $this-polygon-offset-units]} \
	    {set $this-polygon-offset-units 0.0}

	# Set up lights
	global $this-global-light0 # light 0 is the head light
	if {![info exists $this-global-light0]} { set $this-global-light0 1 }
	global $this-global-light1 
	if {![info exists $this-global-light1]} { set $this-global-light1 0 }
	global $this-global-light2 
	if {![info exists $this-global-light2]} { set $this-global-light2 0 }
	global $this-global-light3 
	if {![info exists $this-global-light3]} { set $this-global-light3 0 }
# 	global $this-global-light4 
# 	if {![info exists $this-global-light4]} { set $this-global-light4 0 }
# 	global $this-global-light5 
# 	if {![info exists $this-global-light5]} { set $this-global-light5 0 }
# 	global $this-global-light6
# 	if {![info exists $this-global-light6]} { set $this-global-light6 0 }
# 	global $this-global-light7 
# 	if {![info exists $this-global-light7]} { set $this-global-light7 0 }
	global $this-lightVectors
	if {![info exists $this-lightVectors]} { 
	    set $this-lightVectors \
		[list { 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }]
# 		     { 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }]
	}
	if {![info exists $this-lightColors]} {
	    set $this-lightColors \
		[list {1.0 1.0 1.0} {1.0 1.0 1.0} \
		     {1.0 1.0 1.0} {1.0 1.0 1.0} ]
# 		     {1.0 1.0 1.0} {1.0 1.0 1.0} \
# 		     {1.0 1.0 1.0} {1.0 1.0 1.0} ]
	}

	global $this-sbase
	if {![info exists $this-sbase]} {set $this-sbase 0.4}
	global $this-sr
	if {![info exists $this-sr]} {set $this-sr 1}
	global $this-do_stereo
	if {![info exists $this-do_stereo]} {set $this-do_stereo 0}

	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	set $this-def-color-r 1.0
	set $this-def-color-g 1.0
	set $this-def-color-b 1.0

        # CollabVis code begin
        if {[set $this-have_collab_vis]} {
	    global $this-view_server
	    if {![info exists $this-view_server]} {set $this-view_server 0}
        }
        # CollabVis code end

	global $this-ortho-view
	if {![info exists $this-ortho-view]} { set $this-ortho-view 0 }
    }

    destructor {
    }

    constructor {config} {
	$viewer-c addviewwindow $this
	set_defaults
	init_frame
    }

    method setWindow {w} {
	$this-c listvisuals .standalone

	if {[winfo exists $w]} {
	    destroy $w
	}
	$this-c switchvisual $w 0 640 512
	if {[winfo exists $w]} {
	    bindEvents $w
	}
	$this-c startup
    }

    method bindEvents {w} {
	bind $w <Expose> "$this-c redraw"
	bind $w <Configure> "$this-c redraw"

	bind $w <ButtonPress-1> "$this-c mtranslate start %x %y"
	bind $w <Button1-Motion> "$this-c mtranslate move %x %y"
	bind $w <ButtonRelease-1> "$this-c mtranslate end %x %y"
	bind $w <ButtonPress-2> "$this-c mrotate start %x %y %t"
	bind $w <Button2-Motion> "$this-c mrotate move %x %y %t"
	bind $w <ButtonRelease-2> "$this-c mrotate end %x %y %t"
	bind $w <ButtonPress-3> "$this-c mscale start %x %y"
	bind $w <Button3-Motion> "$this-c mscale move %x %y"
	bind $w <ButtonRelease-3> "$this-c mscale end %x %y"

	bind $w <Control-ButtonPress-1> "$this-c mdolly start %x %y"
	bind $w <Control-Button1-Motion> "$this-c mdolly move %x %y"
	bind $w <Control-ButtonRelease-1> "$this-c mdolly end %x %y"
	bind $w <Control-ButtonPress-2> "$this-c mrotate_eyep start %x %y %t"
	bind $w <Control-Button2-Motion> "$this-c mrotate_eyep move %x %y %t"
	bind $w <Control-ButtonRelease-2> "$this-c mrotate_eyep end %x %y %t"
	bind $w <Control-ButtonPress-3> "$this-c municam start %x %y %t"
	bind $w <Control-Button3-Motion> "$this-c municam move %x %y %t"
	bind $w <Control-ButtonRelease-3> "$this-c municam end %x %y %t"

	bind $w <Shift-ButtonPress-1> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-ButtonPress-2> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-ButtonPress-3> "$this-c mpick start %x %y %s %b"
	bind $w <Shift-Button1-Motion> "$this-c mpick move %x %y %s 1"
	bind $w <Shift-Button2-Motion> "$this-c mpick move %x %y %s 2"
	bind $w <Shift-Button3-Motion> "$this-c mpick move %x %y %s 3"
	bind $w <Shift-ButtonRelease-1> "$this-c mpick end %x %y %s %b"
	bind $w <Shift-ButtonRelease-2> "$this-c mpick end %x %y %s %b"
	bind $w <Shift-ButtonRelease-3> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonPress-1> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-ButtonPress-2> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-ButtonPress-3> "$this-c mpick start %x %y %s %b"
	bind $w <Lock-Button1-Motion> "$this-c mpick move %x %y %s 1"
	bind $w <Lock-Button2-Motion> "$this-c mpick move %x %y %s 2"
	bind $w <Lock-Button3-Motion> "$this-c mpick move %x %y %s 3"
	bind $w <Lock-ButtonRelease-1> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonRelease-2> "$this-c mpick end %x %y %s %b"
	bind $w <Lock-ButtonRelease-3> "$this-c mpick end %x %y %s %b"
    }

    method killWindow { vw } {
        set w .ui[modname]
	if {"$vw"=="$w"} {
	    $this-c killwindow
	}
    }

    method removeMFrame {w} {
	puts EVW:removeMFrame
    }
    
    method addMFrame {w} {
	puts EVW:addMFrame
    }

    method init_frame {} {
	global $this-global-light
	global $this-global-fog
	global $this-global-type
	global $this-global-debug
	global $this-global-clip
	global $this-global-cull
	global $this-global-dl
	global $this-global-movie
	global $this-global-movieName
	global $this-global-movieFrame
	global $this-global-resize
	global $this-x-resize
	global $this-y-resize
	global $this-do_stereo
	global $this-sbase
	global $this-sr
	global $this-do_bawgl
	global $this-tracker_state
	
	set $this-global-light 1
	set $this-global-fog 0
	set $this-global-type Gouraud
	set $this-global-debug 0
	set $this-global-clip 1
	set $this-global-cull 0
	set $this-global-dl 0
	set $this-global-movie 0
	set $this-global-movieName "movie"
	set $this-global-movieFrame 0
	set $this-global-resize 0
	set $this-x-resize 700
	set $this-y-resize 512
	set $this-do_bawgl 0
	set $this-tracker_state 0
    }

    method resize { } {
	puts EVW:resize
    }

    method switch_frames {} {
	puts EVW:switch_frames
    }

    method updatePerf {p1 p2 p3} {
    }

    method switchvisual {idx} {
	set w .ui[modname]
	if {[winfo exists $w.wframe.draw]} {
	    destroy $w.wframe.draw
	}
	$this-c switchvisual $w.wframe.draw $idx 640 512
	if {[winfo exists $w.wframe.draw]} {
	    bindEvents $w.wframe.draw
	    pack $w.wframe.draw -expand yes -fill both
	}
    }	

    method bench {bench} {
	puts EVW:bench
    }

    method makeViewPopup {} {
	puts EVW:makeViewPopup
    }

    method makeSceneMaterialsPopup {} {
	puts EVW:makeSceneMaterialsPopup
    }

    method makeBackgroundPopup {} {
	puts EVW:makeBackgroundPopup
    }

    method updateMode {msg} {
    }   

    method addObject {objid name} {
	global "$this-$objid-light"
	global "$this-$objid-fog"
	global "$this-$objid-type"
	global "$this-$objid-debug"
	global "$this-$objid-clip"
	global "$this-$objid-cull"
	global "$this-$objid-dl"

	set "$this-$objid-type" Default
	set "$this-$objid-light" 1
	set "$this-$objid-fog" 0
	set "$this-$objid-debug" 0
	set "$this-$objid-clip" 0
	set "$this-$objid-cull" 0
	set "$this-$objid-dl" 0

	$this-c autoview
    }

    method addObjectToFrame {objid name frame} {
	puts EVW:addObjectToFrame
    }

    method addObject2 {objid} {
	$this-c autoview
    }
    
    method addObjectToFrame_2 {objid frame} {
	puts EVW:addObjectToFrame_2
    }
    

    method removeObject {objid} {
	puts EVW:removeObject
    }

    method removeObjectFromFrame {objid frame} {
	puts EVW:removeObjectFromFrame
    }

    method makeLineWidthPopup {} {
	puts EVW:makeLineWidthPopup
    }	

    method makePolygonOffsetPopup {} {
	puts EVW:makePolygonOffsetPopup
    }	

    method makePointSizePopup {} {
	puts makePointSizePopup
    }	

    method makeClipPopup {} {
	puts makeClipPopup
    }

    method useClip {} {
	puts useClip
    }

    method setClip {} {
	puts setClip
    }

    method invertClip {} {
	puts invertClip
    }

    method makeAnimationPopup {} {
	puts makeAnimationPopup
    }

    method setFrameRate {rate} {
    }

    method frametime {} {
	puts frametime
    }

    method rstep {} {
	puts rstep
    }

    method rew {} {
	puts rew
    }

    method rplay {} {
	puts rplay
    }

    method play {} {
	puts play
    }

    method step {} {
	puts step
    }

    method ff {} {
	puts ff
    }

    method crap {} {
	puts crap
    }

    method translate {axis amt} {
	puts translate
    }

    method rotate {axis amt} {
	puts rotate
    }

    method rscale {amt} {
	puts rscale
    }

    method zoom {amt} {
	puts zoom
    }

    method pan {amt} {
	puts pan
    }

    method tilt {amt} {
	puts tilt
    }

    method fov {amt} {
	puts fov
    }

    method makeSaveObjectsPopup {} {
	puts makeSaveObjectsPopup
    }

    method doSaveObjects {} {
	puts doSaveObjects
    }

    method do_validate_x {path} {
	puts do_validate_x
    }

    method do_validate_y {path} {
	puts do_validate_y
    }

    method do_aspect {t} {
	puts do_aspect
    }

    method makeLightSources {} {
	puts makeLightSources
    }
	
    method makeLightControl { w i } {
	puts makeLightControl
    }

    method lightColor { w c i } {
	puts lightColor
    }

    method setColor { w c  i color} {
	puts setColor
    }

    method resetLights { w } {
	puts resetLights
    }

    method moveLight { c i x y } {
	puts moveLight
    }

    method lightSwitch {i} {
	puts lightSwitch
    }
	
    method makeSaveImagePopup {} {
	puts makeSaveImagePopup
    }
    
    method changeName { w type} {
	puts changeName
    }

    method doSaveImage {} {
	puts doSaveImage
    }
}

