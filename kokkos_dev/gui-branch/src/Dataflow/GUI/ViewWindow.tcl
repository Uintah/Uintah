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



itcl::class ViewWindow {
    inherit PartGui

    public variable viewer
    public variable w
    public variable renderer

    # parameters to hold current state of detachable part
    protected variable IsAttached 
    protected variable IsDisplayed
    # hold names of detached and attached windows
    protected variable detachedFr
    protected variable attachedFr

    # set defaults values for parameters that weren't set in a script
    public variable saveFile "out.raw"
    public variable saveType "raw"
    
    # Animation parameters
    public variable current_time 0
    public variable tbeg 0
    public variable tend 1
    public variable framerate 15
    public variable totframes 30
    public variable caxes 1
    
    # Need to initialize the background color
    public variable bgcolor_r 0
    public variable bgcolor_g 0
    public variable bgcolor_b 0
    
    public variable sbase 1.0
    public variable sr 1
    public variable do_stereo 0


    variable global_light 1
    variable global_fog 0
    variable global_psize 1
    variable global_type Gouraud
    variable global_debug 0
    variable global_clip 0
    variable global_cull 0
    variable global_dl 0
    variable global_movie 0
    variable global_movieName movie
    variable global_movieFrame 0
    variable global_resize 0
    variable x_resize 640
    variable y_resize 512
    variable do_bawgl 0
    variable tracker_state 0
	
    destructor {
    }

    constructor {} {
    }

    method set-window { win } {
	# for now assume the viewer is global
	global global_viewer
	set viewer $global_viewer

	# set window
	set w $win
	bind $w <Destroy> "$this killWindow %W" 
	wm title $w "ViewWindow"
	wm iconname $w "ViewWindow"
	wm minsize $w 100 100

	# add menus
	frame $w.menu -relief raised -borderwidth 3
	pack $w.menu -fill x
	menubutton $w.menu.renderer -text "Renderer" -underline 0 \
	    -menu $w.menu.renderer.menu
	menu $w.menu.renderer.menu
	
	# Get the list of supported renderers for the pulldown
	set r [$viewer-c listrenderers]
	
	# OpenGL is the preferred renderer, X11 the next best.
	# Otherwise just pick the first one for the default
	if {[lsearch -exact $r OpenGL] != -1} {
	    set renderer OpenGL
	} elseif {[lsearch -exact $r X11] != -1} {
	    set renderer X11
	} else {
	    set renderer [lindex $r 0]
	}
	frame $w.wframe -borderwidth 3 -relief sunken
	pack $w.wframe -expand yes -fill both -padx 4 -pady 4
	
	set width 640
	set height 512
	set wcommand [$this-c setrenderer $renderer $w.wframe.draw \
			  $width $height]
	
	foreach i $r {
	    $w.menu.renderer.menu add radio -label $i \
		-variable [scope renderer] -value $i \
		-command "$this switchRenderer $i"
	}
	menubutton $w.menu.edit -text "Edit" -underline 0 \
	    -menu $w.menu.edit.menu
	menu $w.menu.edit.menu
	$w.menu.edit.menu add command -label "View/Camera..." -underline 0 \
	    -command "$this makeViewPopup"
	$w.menu.edit.menu add command -label "Background..." -underline 0 \
	    -command "$this makeBackgroundPopup"
	$w.menu.edit.menu add command -label "Clipping Planes..." \
	    -underline 0 -command "$this makeClipPopup"
	$w.menu.edit.menu add command -label "Point Size..." -underline 0 \
	    -command "$this makePointSizePopup"
	
	menubutton $w.menu.visual -text "Visual" -underline 0 \
	    -menu $w.menu.visual.menu
	menu $w.menu.visual.menu
	set i 0
	set currentvisual 0
	foreach t [$this-c listvisuals $w] {
	    $w.menu.visual.menu add radiobutton -value $i -label $t \
		-variable [scope currentvisual] \
		-font "-Adobe-Helvetica-bold-R-Normal-*-12-75-*" \
		-command "$this switchvisual $i"
	    incr i
	}
	
	pack $w.menu.visual -side left
	
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
	set mouseModeText $w.bframe.mousemode.text
	$w.bframe.mousemode create text 35 40 -tag mouseModeText \
	    -text " Current Mouse Mode " \
	    -anchor w
	
	frame $w.bframe.v1
	pack $w.bframe.v1 -side left
	button $w.bframe.v1.autoview -text "Autoview" \
	    -command "$this-c autoview"
	pack $w.bframe.v1.autoview -fill x -pady 2 -padx 2
	
	frame $w.bframe.v1.views             
	pack $w.bframe.v1.views -side left -anchor nw -fill x -expand 1
	
	menubutton $w.bframe.v1.views.def -text "   Views   " \
	    -menu $w.bframe.v1.views.def.m -relief raised -padx 2 -pady 2
	
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
	    -variable [scope pos] -value x1_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector -Y" \
	    -variable [scope pos] -value x1_y0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector +Z" \
	    -variable [scope pos] -value x1_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posx add radiobutton -label "Up vector -Z" \
	    -variable [scope pos] -value x1_z0 -command "$this-c Views"
	
	menu $w.bframe.v1.views.def.m.posy
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector +X" \
	    -variable [scope pos] -value y1_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector -X" \
	    -variable [scope pos] -value y1_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector +Z" \
	    -variable [scope pos] -value y1_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posy add radiobutton -label "Up vector -Z" \
	    -variable [scope pos] -value y1_z0 -command "$this-c Views"
	
	menu $w.bframe.v1.views.def.m.posz
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector +X" \
	    -variable [scope pos] -value z1_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector -X" \
	    -variable [scope pos] -value z1_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector +Y" \
	    -variable [scope pos] -value z1_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.posz add radiobutton -label "Up vector -Y" \
	    -variable [scope pos] -value z1_y0 -command "$this-c Views"
	
	menu $w.bframe.v1.views.def.m.negx
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector +Y" \
	    -variable [scope pos] -value x0_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector -Y" \
	    -variable [scope pos] -value x0_y0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector +Z" \
	    -variable [scope pos] -value x0_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negx add radiobutton -label "Up vector -Z" \
	    -variable [scope pos] -value x0_z0 -command "$this-c Views"
	
	menu $w.bframe.v1.views.def.m.negy
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector +X" \
	    -variable [scope pos] -value y0_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector -X" \
	    -variable [scope pos] -value y0_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector +Z" \
	    -variable [scope pos] -value y0_z1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negy add radiobutton -label "Up vector -Z" \
	    -variable [scope pos] -value y0_z0 -command "$this-c Views"
	
	menu $w.bframe.v1.views.def.m.negz
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector +X" \
	    -variable [scope pos] -value z0_x1 -command "$this-c Views" 
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector -X" \
	    -variable [scope pos] -value z0_x0 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector +Y" \
	    -variable [scope pos] -value z0_y1 -command "$this-c Views"
	$w.bframe.v1.views.def.m.negz add radiobutton -label "Up vector -Y" \
	    -variable [scope pos] -value z0_y0 -command "$this-c Views"
	
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
	update
	
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
	
	puts "switchvisual"
	switchvisual 0
	puts "startup"
	$this-c startup
	puts "done"
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
	#	if { [winfo exists $m] } 
	puts "Initializing frame ... "
	
	set r "$this-c redraw"
	bind $m <Double-ButtonPress-1> "$this switch_frames"
	
	label $m.cut -anchor w -text $msg \
	    -font "-Adobe-Helvetica-bold-R-Normal-*-12-75-*"
	pack $m.cut -side top -anchor w -pady 5 -padx 5
	bind $m.cut <Double-ButtonPress-1> "$this switch_frames"
	
	frame $m.eframe
	
	checkbutton $m.eframe.light -text Lighting \
	    -variable [scope global_light] \
	    -command "$this-c redraw"
	checkbutton $m.eframe.fog -text Fog -variable [scope global_fog] \
	    -command "$this-c redraw"
	checkbutton $m.eframe.bbox -text BBox \
	    -variable [scope global_debug] \
	    -command "$this-c redraw"
	checkbutton $m.eframe.clip -text "Use Clip" \
	    -variable [scope global_clip] \
	    -command "$this-c redraw"
	checkbutton $m.eframe.cull -text "Back Cull" \
	    -variable [scope global_cull] \
	    -command "$this-c redraw"
	checkbutton $m.eframe.dl -text "Display List" \
	    -variable [scope global_dl] -command "$this-c redraw"
	
	pack $m.eframe -anchor w -padx 2 -side left
	pack  $m.eframe.light $m.eframe.fog $m.eframe.bbox $m.eframe.clip \
	    $m.eframe.cull $m.eframe.dl -in $m.eframe -side top -anchor w
	
        frame $m.eframe.f -relief groove -borderwidth 2
        pack $m.eframe.f -side top -anchor w
        label $m.eframe.f.l -text "Record Movie as:"
        pack $m.eframe.f.l -side top 
	checkbutton $m.eframe.f.resize -text "Resize: " \
	    -variable [scope global_resize] \
	    -offvalue 0 -onvalue 1 -command "$this resize; $this-c redraw"
	entry $m.eframe.f.e1 -textvariable [scope x_resize] -width 4
	label $m.eframe.f.x -text x
	entry $m.eframe.f.e2 -textvariable [scope y_resize] -width 4
        radiobutton $m.eframe.f.none -text "Stop Recording" \
            -variable [scope global_movie] -value 0 -command "$this-c redraw"
	radiobutton $m.eframe.f.raw -text "Raw Frames" \
            -variable [scope global_movie] -value 1 -command "$this-c redraw"
#	if { [$this-c have_mpeg] } {
#	    radiobutton $m.eframe.f.mpeg -text "Mpeg" \
\#		-variable [scope global_movie] -value 2 \
\#		-command "$this-c redraw"
#	} else {
	    radiobutton $m.eframe.f.mpeg -text "Mpeg" \
		-variable [scope global_movie] -value 2 \
		-state disabled -disabledforeground "" \
		-command "$this-c redraw"
#	}
	entry $m.eframe.f.moviebase -relief sunken -width 12 \
	    -textvariable [scope global_movieName]
        pack $m.eframe.f.none $m.eframe.f.raw $m.eframe.f.mpeg \
            -side top  -anchor w
        pack $m.eframe.f.moviebase -side top -anchor w -padx 2 -pady 2
	pack $m.eframe.f.resize $m.eframe.f.e1 \
	    $m.eframe.f.x $m.eframe.f.e2 -side left  -anchor w
	
	make_labeled_radio $m.shade "Shading:" $r top global_type \
	    {Wire Flat Gouraud}
	pack $m.shade -in $m.eframe -side top -anchor w
	
	frame $m.objlist -relief groove -borderwidth 2
	pack $m.objlist -side left -padx 2 -pady 2 -fill y
	label $m.objlist.title -text "Objects:"
	pack $m.objlist.title -side top
	canvas $m.objlist.canvas -width 400 -height 100 \
	    -scrollregion "0 0 400 100" \
	    -yscrollcommand "$m.objlist.scroll set" \
	    -borderwidth 0 -yscrollincrement 10
	pack $m.objlist.canvas -side right -padx 2 -pady 2 -fill y
	
	frame $m.objlist.canvas.frame -relief sunken -borderwidth 2
	pack $m.objlist.canvas.frame
	$m.objlist.canvas create window 0 1 -window $m.objlist.canvas.frame \
	    -anchor nw
	
	scrollbar $m.objlist.scroll -relief sunken \
	    -command "$m.objlist.canvas yview"
	pack $m.objlist.scroll -fill y -side right -padx 2 -pady 2
	
        checkbutton $m.caxes -text "Show Axes" \
	    -variable [scope caxes] -onvalue 1 -offvalue 0 \
	    -command "$this-c centerGenAxes; $this-c redraw"
	pack $m.caxes -side top
	
	checkbutton $m.stereo -text "Stereo" -variable [scope do_stereo] \
	    -command "$this-c redraw"
	pack $m.stereo -side top
	
	scale $m.sbase -variable [scope sbase] -length 100 -from 0.1 -to 4 \
	    -resolution 0.05 -orient horizontal -label "Fusion Scale:" \
	    -command "$this-c redraw"
	pack $m.sbase -side top
	
	bind $m.eframe.f.e1 <Return> "$this resize"
	bind $m.eframe.f.e2 <Return> "$this resize"
	if {$global_resize == 0} {
	    set color "#505050"
	    $m.eframe.f.x configure -foreground $color
	    $m.eframe.f.e1 configure -state disabled -foreground $color
	    $m.eframe.f.e2 configure -state disabled -foreground $color
	}
    }

    # yarden fixed
    
    method resize { } {
	if { $global_resize == 0 } {
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
	    set xsize $x_resize
	    set ysize $y_resize
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

    #yarden not fixed

    method switch_frames {} {
	if {$IsDisplayed!=0} {
	    
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
    
    method switchRenderer {renderer} {
	set width [winfo width $w.wframe.draw]
	set height [winfo height $w.wframe.draw]
	destroy $w.wframe.draw
	set wcommand [$this-c setrenderer $renderer $w.wframe.draw \
			  $width $height]
	eval $wcommand
	bindEvents $w.wframe.draw
	pack $w.wframe.draw -expand yes -fill both
    }
    
    method updatePerf {p1 p2 p3} {
	$w.bframe.pf.perf1 configure -text $p1
	$w.bframe.pf.perf2 configure -text $p2
	$w.bframe.pf.perf3 configure -text $p3
    }

    method switchvisual {idx} {
	if {[winfo exists $w.wframe.draw]} {
	    destroy $w.wframe.draw
	}
	$this-c switchvisual $w.wframe.draw $idx 640 512
	if {[winfo exists $w.wframe.draw]} {
	    bindEvents $w.wframe.draw
	    pack $w.wframe.draw -expand yes -fill both
	}
    }	
    
    #
    #   yarden: not fixed 
    method bench {bench} {
        upvar #0 $bench b
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
	set wv .view[modname]
	toplevel $wv
	wm title $wv "View"
	wm iconname $wv view
	wm minsize $wv 100 100
	set c "$this-c redraw "
	makePoint $wv.eyep "Eye Point" $view-eyep $c
	pack $wv.eyep -side left -expand yes -fill x
	makePoint $wv.lookat "Look at Point" $view-lookat $c
	pack $wv.lookat -side left -expand yes -fill x
	makeNormalVector $wv.up "Up Vector" $view-up $c
	pack $wv.up -side left -expand yes -fill x
	frame $wv.f -relief groove -borderwidth 2
	pack $wv.f
	scale $wv.f.fov -orient horizontal -variable [scope $view-fov] \
	    -from 0 -to 180 -label "Field of View:" \
	    -showvalue true -tickinterval 90 \
	    -digits 3 \
	    -command $c
	pack $wv.f.fov -expand yes -fill x
	#  	entry $w.f.fove -textvariable $view-fov
	#  	pack $w.f.fove -side top -expand yes -fill x
	#  	bind $w.f.fove <Return> "$command $view-fov"
    }
    
    method makeBackgroundPopup {} {
	set wbg .bg[modname]
	toplevel $wbg
	wm title $wbg "Background"
	wm iconname $wbg background
	wm minsize $wbg 100 100
	set c "$this-c redraw "
	makeColorPicker $wbg bgcolor $c ""
    }
    
    #
    # yarden: fixed
    method updateMode {msg} {
	global .ui[modname].bframe.mousemode
	set mouseModeText $w.bframe.mousemode
	$mouseModeText itemconfigure mouseModeText -text $msg
    }   
    
    method addObject {objid name} {
	addObjectToFrame $objid $name $detachedFr
	addObjectToFrame $objid $name $attachedFr
    }
    
    method addObjectToFrame {objid name frame} {
	set m $frame.f
	frame  $m.objlist.canvas.frame.objt$objid
	checkbutton $m.objlist.canvas.frame.obj$objid -text $name \
	    -relief flat -variable [scope $name] -command "$this-c redraw"
	
	set newframeheight [winfo reqheight $m.objlist.canvas.frame.obj$objid]
	
	set menun $m.objlist.canvas.frame.menu$objid.menu
	
	menubutton $m.objlist.canvas.frame.menu$objid -text Shading \
	    -relief raised -menu $menun
	menu $menun
	$menun add checkbutton -label Lighting -variable [scope $objid-light] \
	    -command "$this-c redraw"
	$menun add checkbutton -label BBox -variable [scope $objid-debug] \
	    -command "$this-c redraw"
	$menun add checkbutton -label Fog -variable [scope $objid-fog] \
	    -command "$this-c redraw"
	$menun add checkbutton -label "Use Clip" -variable [scope $objid-clip]\
	    -command "$this-c redraw"
	$menun add checkbutton -label "Back Cull" \
	    -variable [scope $objid-cull] -command "$this-c redraw"
	$menun add checkbutton -label "Display List" \
	    -variable [scope $objid-dl] -command "$this-c redraw"
	
	set $objid-type Default
	set $objid-light 1
	set $objid-fog 0
	set $objid-debug 0
	set $objid-clip 0
	set $objid-cull 0
	set $objid-dl 0
	
	set menuvar  $m.objlist.canvas.frame.menu2_$objid
	set menup [tk_optionMenu $menuvar $objid-type Wire Flat Gouraud Default]
	
	$menup entryconfigure 0 -command "[$menup entrycget 0 -command] ; $this-c redraw"
	$menup entryconfigure 1 -command "[$menup entrycget 1 -command] ; $this-c redraw"
	$menup entryconfigure 2 -command "[$menup entrycget 2 -command] ; $this-c redraw"
	$menup entryconfigure 3 -command "[$menup entrycget 3 -command] ; $this-c redraw"
	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid $m.objlist.canvas.frame.menu2_$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
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
	set m $frame.f
	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid $m.objlist.canvas.frame.menu2_$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
    }
    
    
    method removeObject {objid} {
	removeObjectFromFrame $objid $detachedFr
	removeObjectFromFrame $objid $attachedFr
    }
    
    method removeObjectFromFrame {objid frame} {
	set m $frame.f
	pack forget $m.objlist.canvas.frame.objt$objid
    }
    
    #
    #   yarden: not fixed
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
	global $this-global-psize
	scale $w.f.scale -command "$this-c redraw" -variable \
	    $this-global-psize -orient horizontal -from 1 -to 5 \
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
    method makeSaveImagePopup {} {
	global $this-saveFile
	global $this-saveType
	toplevel .ui[modname]-saveImage
	set w .ui[modname]-saveImage
	makeFilebox $w \
	    $this-saveFile "$this doSaveImage" \
	    "destroy $w"
	#$w.f.sel.sel configure -textvariable $saveFile
	set ex $w.f.extra
	radiobutton $ex.raw -variable $this-saveType \
	    -text "Raw File" -value "raw" \
	    -command "$this changeName $w raw"
	pack $ex.raw -side top -anchor w
	set sgi [$this-c sgi_defined]
	if { $sgi == 1 || $sgi == 2 } {
	    radiobutton $ex.rgb -variable $this-saveType \
		-text "SGI RGB File" -value "rgb" \
		-command "$this changeName $w rgb"
	    radiobutton $ex.ppm -variable $this-saveType \
		-text "PPM File" -value "ppm" \
		-command "$this changeName $w ppm"
	    radiobutton $ex.jpg -variable $this-saveType \
		-text "JPEG File" -value "jpg" \
		-command "$this changeName $w jpg"
	} else {
	    radiobutton $ex.rgb -variable $this-saveType \
		-text "SGI RGB File" -value "rgb" \
		-state disabled -disabledforeground ""
	    radiobutton $ex.ppm -variable $this-saveType \
		-text "PPM File" -value "ppm" \
		-state disabled -disabledforeground ""

	    radiobutton $ex.jpg -variable $this-saveType \
		-text "JPEG File" -value "jpg" \
		-state disabled -disabledforeground ""
	}
	
	if { [set $this-saveType] == "rgb" } { 
	    $ex.rgb select 
	} elseif { [set $this-saveType] == "ppm" } {
	    $ex.ppm select
	} elseif { [set $this-saveType] == "jpg" } {
	    $ex.jpg select
	} else { $ex.raw select }
	pack $ex.rgb $ex.ppm $ex.jpg -side top -anchor w
    }

    #
    # yarden: fixed
    method changeName { w type} {
	set name [split $saveFile .]
	set newname [lreplace $name end end $type]
	set saveFile [join $newname .]
    }
    method doSaveImage {} {
	$this-c dump_viewwindow $saveFile $saveType
    }
}
