
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
	menubutton $w.menu.renderer -text "Renderer" -underline 0 \
		-menu $w.menu.renderer.menu
	menu $w.menu.renderer.menu
	set r [$salmon-c listrenderers]
	
	# OpenGL is the preferred renderer, X11 the next best.
	# Otherwise just pick the first one
	global $this-renderer
	if {[lsearch -exact $r OpenGL] != -1} {
	    set $this-renderer OpenGL
	} elseif {[lsearch -exact $r X11] != -1} {
	    set $this-renderer X11
	} else {
	    set $this-renderer [lindex $r 0]
	}
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
	$w.menu.edit.menu add command -label "Clipping Planes..." -underline 0
	menubutton $w.menu.spawn -text "Spawn" -underline 0 \
		-menu $w.menu.spawn.menu
	menu $w.menu.spawn.menu
	$w.menu.spawn.menu add command -label "Spawn Independent..." -underline 6
	$w.menu.spawn.menu add command -label "Spawn Child..." -underline 6
	menubutton $w.menu.dialbox -text "Dialbox" -underline 0 \
		-menu $w.menu.dialbox.menu
	menu $w.menu.dialbox.menu
	$w.menu.dialbox.menu add command -label "Translate/Scale..." -underline 0 \
		-command "$this dialbox_ts"
	pack $w.menu.edit -side left
	pack $w.menu.renderer -side left
	pack $w.menu.spawn -side left
	pack $w.menu.dialbox -side left
	tk_menuBar $w.menu $w.menu.edit $w.menu.renderer \
		$w.menu.spawn $w.menu.dialbox
	
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
		-font "-Adobe-Helvetica-bold-R-Normal-*-140-75-*" \
		-command "$this addMFrame $w"
	pack $w.bframe.more -pady 2 -padx 2 -anchor se -side right
	
	set m $w.mframe.f
	set r "$this-c redraw"
	make_labeled_radio $m.shade "Shading:" $r top $this-shading \
		{Wire Flat Gouraud Phong}
	pack $m.shade -anchor w -padx 2 -side left
	global $this-shading
	set $this-shading Phong
	
	frame $m.objlist -relief groove -borderwidth 2
	pack $m.objlist -side left -padx 2 -pady 2 -fill y
	label $m.objlist.title -text "Objects:"
	pack $m.objlist.title -side top
	canvas $m.objlist.canvas -width 400 -height 100 \
		-yscroll "$m.objlist.scroll set" -borderwidth 0
	pack $m.objlist.canvas -side left -padx 2 -pady 2 -fill y
	
	frame $m.objlist.canvas.frame -relief sunken -borderwidth 2
	pack $m.objlist.canvas.frame
	$m.objlist.canvas create window 0 0 -window $m.objlist.canvas.frame \
		-anchor nw
	
	scrollbar $m.objlist.scroll -relief sunken \
		-command "$m.objlist.canvas yview"
	pack $m.objlist.scroll -fill y -side right -padx 2 -pady 2
	
	frame $w.wframe -borderwidth 3 -relief sunken
	pack $w.wframe -expand yes -fill both -padx 4 -pady 4
	
	set width 600
	set height 500
	set wcommand [$this-c setrenderer [set $this-renderer] $w.wframe.draw $width $height]
	eval $wcommand
	bindEvents $w.wframe.draw
	pack $w.wframe.draw -expand yes -fill both
	
	$this-c startup
    }
    method bindEvents {w} {
	bind $w <Expose> "$this-c redraw"
	bind $w <ButtonPress-1> "$this-c mtranslate start %x %y"
	bind $w <Button1-Motion> "$this-c mtranslate move %x %y"
	bind $w <ButtonRelease-1> "$this-c mtranslate end %x %y"
	bind $w <ButtonPress-2> "$this-c mrotate start %x %y"
	bind $w <Button2-Motion> "$this-c mrotate move %x %y"
	bind $w <ButtonRelease-2> "$this-c mrotate end %x %y"
	bind $w <ButtonPress-3> "$this-c mscale start %x %y"
	bind $w <Button3-Motion> "$this-c mscale move %x %y"
	bind $w <ButtonRelease-3> "$this-c mscale end %x %y"
	bind $w <Shift-ButtonPress-1> "$this-c mpick start %x %y"
	bind $w <Shift-Button1-Motion> "$this-c mpick move %x %y"
	bind $w <Shift-ButtonRelease-1> "$this-c mpick end %x %y"
    }
    method removeMFrame {w} {
	pack forget $w.mframe.f
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
	checkbutton $m.objlist.canvas.frame.obj$objid -text $name \
		-relief flat -variable "$this-$name" -command "$this-c redraw"
	pack $m.objlist.canvas.frame.obj$objid -side top -anchor w
    }
    
    method addObject2 {objid} {
	set w .ui$this
	set m $w.mframe.f
	pack $m.objlist.canvas.frame.obj$objid -side top -anchor w
    }
    
    method removeObject {objid} {
	set w .ui$this
	set m $w.mframe.f
	pack forget $m.objlist.canvas.frame.obj$objid
    }
}
