
catch {rename Salmon ""} 

itcl_class Salmon {
    inherit Module
    constructor {config} {
	set name Salmon
	set_defaults
	set make_progress_graph 0
	set make_time 0
    }
    method set_defaults {} {
    }
    protected nextrid 0
    method makeRoeID {} {
	set id $this-Roe_$nextrid
	incr nextrid
	return $id
    }
    protected roe
    method ui {} {
	set rid [makeRoeID]
	Roe $rid -salmon $this
	lappend roe $rid
    }
}

catch {rename Roe ""}

itcl_class Roe {
    public salmon
    constructor {config} {
	$salmon-c addroe $this
	set w .ui$this
	toplevel $w
	wm title $w "Roe"
	wm iconname $w "Roe"
	wm minsize $w 100 100
	frame $w.menu -relief raised -borderwidth 3
	pack $w.menu -fill x
	menubutton $w.menu.file -text "File" -underline 0 \
		-menu $w.menu.file.menu
	menu $w.menu.file.menu
	$w.menu.file.menu add command -label "Save geom file..." -underline 0 \
		-command "$this makeSaveObjectsPopup"
	$w.menu.file.menu add command -label "Save image file..." -underline 0 \
		-command "$this makeSaveImagePopup"
	menubutton $w.menu.renderer -text "Renderer" -underline 0 \
		-menu $w.menu.renderer.menu
	menu $w.menu.renderer.menu
	# Animation parameters
	global $this-current_time
	set $this-current_time 0
	global $this-tbeg
	set $this-tbeg 0
	global $this-tend
	set $this-tend 1
	global $this-framerate
	set $this-framerate 15
	global $this-totframes
	set $this-totframes 30
	#
	# Get the list of supported renderers for the pulldown
	#
	set r [$salmon-c listrenderers]
	
	# OpenGL is the preferred renderer, X11 the next best.
	# Otherwise just pick the first one for the default
	global $this-renderer
	if {[lsearch -exact $r OpenGL] != -1} {
	    set $this-renderer OpenGL
	} elseif {[lsearch -exact $r X11] != -1} {
	    set $this-renderer X11
	} else {
	    set $this-renderer [lindex $r 0]
	}
	frame $w.wframe -borderwidth 3 -relief sunken
	pack $w.wframe -expand yes -fill both -padx 4 -pady 4
	
	set width 640
	set height 512
	set wcommand [$this-c setrenderer [set $this-renderer] $w.wframe.draw $width $height]

	foreach i $r {
	    $w.menu.renderer.menu add radio -label $i -variable $this-renderer \
		    -value $i -command "$this switchRenderer $i"
	}
	menubutton $w.menu.edit -text "Edit" -underline 0 \
		-menu $w.menu.edit.menu
	menu $w.menu.edit.menu
	$w.menu.edit.menu add command -label "View/Camera..." -underline 0 \
		-command "$this makeViewPopup"
	$w.menu.edit.menu add command -label "Renderer..." -underline 0
	$w.menu.edit.menu add command -label "Materials..." -underline 0
	$w.menu.edit.menu add command -label "Light Sources..." -underline 0
	$w.menu.edit.menu add command -label "Background..." -underline 0 \
		-command "$this makeBackgroundPopup"
	$w.menu.edit.menu add command -label "Clipping Planes..." -underline 0 -command "$this makeClipPopup"
	$w.menu.edit.menu add command -label "Animation..." -underline 0 \
		-command "$this makeAnimationPopup"
	$w.menu.edit.menu add command -label "Point Size..." -underline 0 \
		-command "$this makePointSizePopup"
	menubutton $w.menu.spawn -text "Spawn" -underline 0 \
		-menu $w.menu.spawn.menu
	menu $w.menu.spawn.menu
	$w.menu.spawn.menu add command -label "Spawn Independent..." -underline 6
	$w.menu.spawn.menu add command -label "Spawn Child..." -underline 6
	menubutton $w.menu.dialbox -text "Dialbox" -underline 0 \
		-menu $w.menu.dialbox.menu
	menu $w.menu.dialbox.menu
	$w.menu.dialbox.menu add command -label "Translate/Scale..." -underline 0 \
		-command "$w.dialbox connect"
	$w.menu.dialbox.menu add command -label "Camera..." -underline 0 \
		-command "$w.dialbox2 connect"

	menubutton $w.menu.visual -text "Visual" -underline 0 \
	    -menu $w.menu.visual.menu
	menu $w.menu.visual.menu
	set i 0
	global $this-currentvisual
	set $this-currentvisual 0
	foreach t [$this-c listvisuals $w] {
	    $w.menu.visual.menu add radiobutton -value $i -label $t \
		-variable $this-currentvisual \
		-font "-adobe-helvetica-bold-r-normal-*-*-80-75-*-*-*-*-*" \
		-command "$this switchvisual $i"
	    incr i
	}

	pack $w.menu.file -side left
	pack $w.menu.edit -side left
	pack $w.menu.renderer -side left
	pack $w.menu.spawn -side left
	pack $w.menu.dialbox -side left
	pack $w.menu.visual -side left
	tk_menuBar $w.menu $w.menu.edit $w.menu.renderer \
		$w.menu.spawn $w.menu.dialbox $w.menu.visual

	# Create Dialbox and attach to it
	Dialbox $w.dialbox "Salmon - Translate/Scale"
	$w.dialbox unbounded_dial 0 "Translate X" 0.0 1.0 "$this translate x"
	$w.dialbox unbounded_dial 2 "Translate Y" 0.0 1.0 "$this translate y" 
	$w.dialbox unbounded_dial 4 "Translate Z" 0.0 1.0 "$this translate z"
	$w.dialbox wrapped_dial 1 "Rotate X" 0.0 0.0 360.0 1.0 "$this rotate x"
	$w.dialbox wrapped_dial 3 "Rotate Y" 0.0 0.0 360.0 1.0 "$this rotate y"
	$w.dialbox wrapped_dial 5 "Rotate Z" 0.0 0.0 360.0 1.0 "$this rotate z"
	$w.dialbox bounded_dial 6 "Scale" 1.0 [expr 1.0/1000.0] 1000.0 1.0 "$this scale"
	
	# Create Dialbox2 and attach to it
	Dialbox $w.dialbox2 "Salmon - Camera"
	$w.dialbox2 bounded_dial 0 "Zoom" 0.0 0.0 1000.0 100.0 "$this zoom"
	$w.dialbox2 wrapped_dial 1 "Pan" 0.0 0.0 360.0 1.0 "$this pan" 
	$w.dialbox2 wrapped_dial 2 "Tilt" 0.0 0.0 360.0 1.0 "$this tilt"
	$w.dialbox2 bounded_dial 3 "FOV" 0.0 0.0 180.0 1.0 "$this fov"

	frame $w.mframe
	frame $w.mframe.f
	pack $w.mframe -side bottom -fill x
	
	frame $w.bframe
	pack $w.bframe -side bottom -fill x
	frame $w.bframe.pf
	pack $w.bframe.pf -side left -anchor n
	label $w.bframe.pf.perf1 -width 32 -text "100000 polygons in 12.33 seconds"
	pack $w.bframe.pf.perf1 -side top -anchor n
	label $w.bframe.pf.perf2 -width 32 -text "Hello"
	pack $w.bframe.pf.perf2 -side top -anchor n
	label $w.bframe.pf.perf3 -width 32 -text "Hello"
	pack $w.bframe.pf.perf3 -side top -anchor n
	canvas $w.bframe.mousemode -width 200 -height 70 -relief groove -borderwidth 2
	pack $w.bframe.mousemode -side left -fill y -pady 2 -padx 2
	frame $w.bframe.v
	pack $w.bframe.v -side left
	button $w.bframe.v.autoview -text "Autoview" -command "$this-c autoview"
	pack $w.bframe.v.autoview -fill x -pady 2
	button $w.bframe.v.sethome -text "Set Home View" -padx 2 \
		-command "$this-c sethome"
	pack $w.bframe.v.sethome -fill x -pady 2
	button $w.bframe.v.gohome -text "Go home" -command "$this-c gohome"
	pack $w.bframe.v.gohome -fill x -pady 2
	frame $w.bframe.dolly
	pack $w.bframe.dolly -side left -anchor nw 
	button $w.bframe.dolly.in -text "In" -command "$this-c dolly .8"
	button $w.bframe.dolly.out -text "Out" -command "$this-c dolly 1.25"
	pack $w.bframe.dolly.in $w.bframe.dolly.out -fill x -padx 2 -pady 2 \
		-anchor nw
	
	button $w.bframe.more -text "+" -padx 3 \
		-font "-Adobe-Helvetica-bold-R-Normal-*-120-75-*" \
		-command "$this addMFrame $w"
	pack $w.bframe.more -pady 2 -padx 2 -anchor se -side right

	set m $w.mframe.f
	set r "$this-c redraw"
	
	frame $m.eframe
	checkbutton $m.eframe.light -text Lighting -variable $this-global-light \
		-command "$this-c redraw"
	checkbutton $m.eframe.fog -text Fog -variable $this-global-fog \
		-command "$this-c redraw"
	checkbutton $m.eframe.bbox -text BBox -variable $this-global-debug \
		-command "$this-c redraw"
	checkbutton $m.eframe.clip -text "Use Clip" -variable $this-global-clip \
		-command "$this-c redraw"
	pack $m.eframe -anchor w -padx 2 -side left
	pack  $m.eframe.light $m.eframe.fog $m.eframe.bbox $m.eframe.clip -in $m.eframe \
		-side top -anchor w
	make_labeled_radio $m.shade "Shading:" $r top $this-global-type \
		{Wire Flat Gouraud}
	pack $m.shade -in $m.eframe -side top -anchor w

	global "$this-global-light"
	global "$this-global-fog"
	global "$this-global-psize"
	global "$this-global-type"
	global "$this-global-debug"
	global "$this-global-clip"

	set "$this-global-light" 1
	set "$this-global-fog" 0
	set "$this-global-psize" 1
	set "$this-global-type" Gouraud
	set "$this-global-debug" 0
	set "$this-global-clip" 0

	frame $m.objlist -relief groove -borderwidth 2
	pack $m.objlist -side left -padx 2 -pady 2 -fill y
	label $m.objlist.title -text "Objects:"
	pack $m.objlist.title -side top
	canvas $m.objlist.canvas -width 400 -height 100 \
	        -scrollregion "0 0 400 100" \
		-yscrollcommand "$m.objlist.scroll set" -borderwidth 0
	pack $m.objlist.canvas -side right -padx 2 -pady 2 -fill y
	
	frame $m.objlist.canvas.frame -relief sunken -borderwidth 2
	pack $m.objlist.canvas.frame
	$m.objlist.canvas create window 0 1 -window $m.objlist.canvas.frame \
		-anchor nw
	
	scrollbar $m.objlist.scroll -relief sunken \
		-command "$m.objlist.canvas yview"
	pack $m.objlist.scroll -fill y -side right -padx 2 -pady 2
	
	global $this-do_stereo
	set $this-do_stereo 0
	checkbutton $m.stereo -text "Stereo" -variable $this-do_stereo
	pack $m.stereo -side top

	global $this-tracker_state
	set $this-tracker_state 0
	checkbutton $m.tracker -text "Tracker" -variable $this-tracker_state \
		-command "$this-c tracker"
	pack $m.tracker -side top

	button $m.tracker_reset -text "Reset tracker" \
		-command "$this-c reset_tracker"
	pack $m.tracker_reset -side top
	
	switchvisual 0
	$this-c startup
    }
    method bindEvents {w} {
	bind $w <Expose> "$this-c redraw"
	bind $w <ButtonPress-1> "$this-c mtranslate start %x %y"
	bind $w <Button1-Motion> "$this-c mtranslate move %x %y"
	bind $w <ButtonRelease-1> "$this-c mtranslate end %x %y"
	bind $w <ButtonPress-2> "$this-c mrotate start %x %y %t"
	bind $w <Button2-Motion> "$this-c mrotate move %x %y %t"
	bind $w <ButtonRelease-2> "$this-c mrotate end %x %y %t"
	bind $w <ButtonPress-3> "$this-c mscale start %x %y"
	bind $w <Button3-Motion> "$this-c mscale move %x %y"
	bind $w <ButtonRelease-3> "$this-c mscale end %x %y"
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
    method removeMFrame {w} {
	pack forget $w.mframe.f
	$w.mframe config -height 1
	$w.bframe.more configure -command "$this addMFrame $w" -text "+"
    }

    method addMFrame {w} {
	pack $w.mframe.f -anchor w
	$w.bframe.more configure -command "$this removeMFrame $w" -text "-"
    }

    method switchRenderer {renderer} {
	set w .ui$this
	set width [winfo width $w.wframe.draw]
	set height [winfo height $w.wframe.draw]
	destroy $w.wframe.draw
	set wcommand [$this-c setrenderer [set $this-renderer] $w.wframe.draw $width $height]
	eval $wcommand
	bindEvents $w.wframe.draw
	pack $w.wframe.draw -expand yes -fill both
    }

    method updatePerf {p1 p2 p3} {
	set w .ui$this
	$w.bframe.pf.perf1 configure -text $p1
	$w.bframe.pf.perf2 configure -text $p2
	$w.bframe.pf.perf3 configure -text $p3
    }

    method switchvisual {idx} {
	set w .ui$this
	if {[winfo exists $w.wframe.draw]} {
	    destroy $w.wframe.draw
	}
	$this-c switchvisual $w.wframe.draw $idx 640 512
	if {[winfo exists $w.wframe.draw]} {
	    bindEvents $w.wframe.draw
	    pack $w.wframe.draw -expand yes -fill both
	}
    }	

    method makeViewPopup {} {
	set w .view$this
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
		-showvalue true -tickinterval 30 \
		-digits 3 \
		-command $c
	pack $w.f.fov -expand yes -fill x
    }

    method makeBackgroundPopup {} {
	set w .bg$this
	toplevel $w
	wm title $w "Background"
	wm iconname $w background
	wm minsize $w 100 100
	set c "$this-c redraw "
	makeColorPicker $w $this-bgcolor $c ""
    }

    method updateMode {msg} {
    }   

    method addObject {objid name} {
	set w .ui$this
	set m $w.mframe.f
	frame  $m.objlist.canvas.frame.objt$objid
	checkbutton $m.objlist.canvas.frame.obj$objid -text $name \
		-relief flat -variable "$this-$name" -command "$this-c redraw"
	
	set menun $m.objlist.canvas.frame.menu$objid.menu

	menubutton $m.objlist.canvas.frame.menu$objid -text Shading \
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

	global "$this-$objid-light"
	global "$this-$objid-fog"
	global "$this-$objid-type"
	global "$this-$objid-debug"
	global "$this-$objid-clip"

	set "$this-$objid-type" Default
	set "$this-$objid-light" 1
	set "$this-$objid-fog" 0
	set "$this-$objid-debug" 0
	set "$this-$objid-clip" 0

	set menuvar  $m.objlist.canvas.frame.menu2_$objid
	set menup [tk_optionMenu $menuvar $this-$objid-type Wire Flat Gouraud Default]

	$menup entryconfigure 0 -command "[$menup entrycget 0 -command] ; $this-c redraw"
	$menup entryconfigure 1 -command "[$menup entrycget 1 -command] ; $this-c redraw"
	$menup entryconfigure 2 -command "[$menup entrycget 2 -command] ; $this-c redraw"
	$menup entryconfigure 3 -command "[$menup entrycget 3 -command] ; $this-c redraw"
	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid $m.objlist.canvas.frame.menu2_$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
	#tkwait visibility $m.objlist.canvas.frame.obj$objid
	update idletasks
	set width [winfo width $m.objlist.canvas.frame]
	set height [winfo height $m.objlist.canvas.frame]
	$m.objlist.canvas configure -scrollregion "0 0 $width $height"
	set view [$m.objlist.canvas yview]
	$m.objlist.scroll set [lindex $view 0] [lindex $view 1]
    }
    
    method addObject2 {objid} {
	set w .ui$this
	set m $w.mframe.f
	pack $m.objlist.canvas.frame.objt$objid -side top -anchor w
	pack $m.objlist.canvas.frame.obj$objid  $m.objlist.canvas.frame.menu$objid $m.objlist.canvas.frame.menu2_$objid -in $m.objlist.canvas.frame.objt$objid -side left -anchor w
    }
    
    method removeObject {objid} {
	set w .ui$this
	set m $w.mframe.f
	pack forget $m.objlist.canvas.frame.objt$objid
    }

    method makePointSizePopup {} {
	set w .psize$this
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
	set w .clip$this
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
	set $clip-normal-d [set $clip-normal-d-$cs]
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
	set w .anim$this
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
		-from 0 -to 10 -label "Begin time:" \
		-resolution 0.01 -digits 4 \
		-showvalue true -tickinterval 2
	scale $w.tend -orient horizontal -variable $this-tend \
		-from 0 -to 10 -label "End time:" \
		-resolution 0.01 -digits 4 \
		-showvalue true -tickinterval 2
	scale $w.ct -orient horizontal -variable $this-current_time \
		-from 0 -to 10 -label "Current time:" \
		-resolution 0.01 -digits 4 \
		-showvalue true -tickinterval 2
	pack $w.tbeg $w.tend $w.ct -side top -fill x
	entry $w.savefile -textvariable $this-saveprefix
	pack $w.savefile -side top -fill x
    }
    method setFrameRate {rate} {
	set w .anim$this
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
	toplevel .ui$this-save
	global $this-saveobjfile $this-saveformat
	set $this-saveobjfile "out.geom"
	makeFilebox .ui$this-save $this-saveobjfile \
		"$this doSaveObjects" "destroy .ui$this-save"
	set ex .ui$this-save.f.extra
	radiobutton $ex.geomb -variable $this-saveformat \
		-text "SCIRun geom object file (Binary)" -value "scirun_binary"
	radiobutton $ex.geoma -variable $this-saveformat \
		-text "SCIRun geom object file (ASCII)" -value "scirun_ascii"
	radiobutton $ex.vrml -variable $this-saveformat \
		-text "VRML file" -value "vrml"
	$ex.geomb select
	pack $ex.geomb $ex.geoma $ex.vrml -side top -anchor w
    }
    method doSaveObjects {} {
	global $this-saveobjfile $this-saveformat
	$this-c saveobj [set $this-saveobjfile] [set $this-saveformat]
    }
}
