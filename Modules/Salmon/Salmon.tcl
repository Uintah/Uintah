
proc uiSalmon {modid} {
    global nextrid,$modid
    set nextrid,$modid 0
    spawnIndependentRoe $modid
}

proc makeRoeID {modid} {
    global nextrid,$modid
    set n [set nextrid,$modid]
    set rid roe_$n,$modid
    incr nextrid,$modid
    return $rid
}

proc spawnIndependentRoe {modid} {
    set rid [makeRoeID $modid]
    makeRoe $modid $rid
}

proc makeRoe {salmon rid} {
    $salmon addroe $rid
    set w .$rid
    toplevel $w
    wm title $w "Roe"
    wm iconname $w "Roe"
    wm minsize $w 100 100
    frame $w.menu -relief raised -borderwidth 3
    pack $w.menu -fill x
    menubutton $w.menu.renderer -text "Renderer" -underline 0 \
	    -menu $w.menu.renderer.menu
    menu $w.menu.renderer.menu
    set r [$salmon listrenderers]
    global renderer$rid
    # OpenGL is the preferred renderer, X11 the next best.
    # Otherwise just pick the first one
    if {[lsearch -exact $r OpenGL] != -1} {
	set renderer OpenGL
    } elseif {[lsearch -exact $r X11] != -1} {
	set renderer X11
    } else {
	set renderer [lindex $r 0]
    }
    set renderer$rid $renderer
    foreach i $r {
	$w.menu.renderer.menu add radio -label $i -variable renderer$rid \
	    -value $i -command "switchRenderer $rid $i"
    }
    menubutton $w.menu.edit -text "Edit" -underline 0 \
	    -menu $w.menu.edit.menu
    menu $w.menu.edit.menu
    $w.menu.edit.menu add command -label "View/Camera..." -underline 0 \
	    -command "makeViewPopup $rid"
    $w.menu.edit.menu add command -label "Renderer..." -underline 0
    $w.menu.edit.menu add command -label "Materials..." -underline 0
    $w.menu.edit.menu add command -label "Light Sources..." -underline 0
    $w.menu.edit.menu add command -label "Background..." -underline 0 \
	    -command "makeBackgroundPopup $rid"
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
	    -command "dialbox_ts $rid"
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
    canvas $w.bframe.mousemode -width 200 -height 70 -relief groove -borderwidth 2
    pack $w.bframe.mousemode -side left -fill y -pady 2 -padx 2
    frame $w.bframe.v
    pack $w.bframe.v -side left
    button $w.bframe.v.autoview -text "Autoview" -command "$rid autoview"
    pack $w.bframe.v.autoview -fill x -pady 2
    button $w.bframe.v.sethome -text "Set Home View" -padx 2 \
	    -command "$rid sethome"
    pack $w.bframe.v.sethome -fill x -pady 2
    button $w.bframe.v.gohome -text "Go home" -command "$rid gohome"
    pack $w.bframe.v.gohome -fill x -pady 2
    frame $w.bframe.dolly
    pack $w.bframe.dolly -side left -anchor nw 
    button $w.bframe.dolly.in -text "In" -command "$rid dolly .8"
    button $w.bframe.dolly.out -text "Out" -command "$rid dolly 1.25"
    pack $w.bframe.dolly.in $w.bframe.dolly.out -fill x -padx 2 -pady 2 \
	    -anchor nw

    button $w.bframe.more -text "+" -padx 3 \
	    -font "-Adobe-Helvetica-bold-R-Normal-*-140-75-*" \
	    -command "addMFrame $w"
    pack $w.bframe.more -pady 2 -padx 2 -anchor se -side right

    set r "$rid redraw"
    set m $w.mframe.f
    frame $m.shade -borderwidth 2 -relief groove
    pack $m.shade -anchor w -padx 2 -side left
    label $m.shade.title -text "Shading:" -anchor w -relief flat 
    pack $m.shade.title -fill x -padx 2
    radiobutton $m.shade.wire -text "Wire" -anchor w -relief flat \
	    -variable shading,$rid -command $r
    pack $m.shade.wire -fill x -padx 2
    radiobutton $m.shade.flat -text "Flat" -anchor w -relief flat \
	    -variable shading,$rid -command $r
    pack $m.shade.flat -fill x -padx 2
    radiobutton $m.shade.gouraud -text "Gouraud" -anchor w -relief flat \
	    -padx 2 -variable shading,$rid -command $r
    pack $m.shade.gouraud -fill x -padx 2
    radiobutton $m.shade.phong -text "Phong" -anchor w -relief flat \
	    -variable shading,$rid -command $r
    pack $m.shade.phong -fill x -padx 2
    $m.shade.phong select


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
    set wcommand [$rid setrenderer $renderer $w.wframe.draw $width $height]
    eval $wcommand
    bindEvents $w.wframe.draw $rid
    pack $w.wframe.draw -expand yes -fill both

    $rid startup
}

proc bindEvents {w rid} {
    bind $w <Expose> "$rid redraw"
    bind $w <ButtonPress-1> "$rid mtranslate start %x %y"
    bind $w <Button1-Motion> "$rid mtranslate move %x %y"
    bind $w <ButtonRelease-1> "$rid mtranslate end %x %y"
    bind $w <ButtonPress-2> "$rid mrotate start %x %y"
    bind $w <Button2-Motion> "$rid mrotate move %x %y"
    bind $w <ButtonRelease-2> "$rid mrotate end %x %y"
    bind $w <ButtonPress-3> "$rid mscale start %x %y"
    bind $w <Button3-Motion> "$rid mscale move %x %y"
    bind $w <ButtonRelease-3> "$rid mscale end %x %y"
    bind $w <Shift-ButtonPress-1> "$rid mpick start %x %y"
    bind $w <Shift-Button1-Motion> "$rid mpick move %x %y"
    bind $w <Shift-ButtonRelease-1> "$rid mpick end %x %y"
}

proc removeMFrame {w} {
    pack forget $w.mframe.f
    $w.bframe.more configure -command "addMFrame $w" -text "+"
}

proc addMFrame {w} {
    pack $w.mframe.f -anchor w
    $w.bframe.more configure -command "removeMFrame $w" -text "-"
}

proc switchRenderer {rid renderer} {
    set w .$rid
    set width [winfo width $w.wframe.draw]
    set height [winfo height $w.wframe.draw]
    destroy $w.wframe.draw
    set wcommand [$rid setrenderer $renderer $w.wframe.draw $width $height]
    eval $wcommand
    bindEvents $w.wframe.draw $rid
    pack $w.wframe.draw -expand yes -fill both
}

proc updatePerf {rid p1 p2} {
    set w .$rid
    $w.bframe.pf.perf1 configure -text $p1
    $w.bframe.pf.perf2 configure -text $p2
}

proc makeViewPopup {rid} {
    set w .view$rid
    toplevel $w
    wm title $w "View"
    wm iconname $w view
    wm minsize $w 100 100
    set c "$rid redraw "
    set view view,$rid
    makePoint $w.eyep "Eye Point" eyep,$view $c
    pack $w.eyep -side left -expand yes -fill x
    makePoint $w.lookat "Look at Point" lookat,$view $c
    pack $w.lookat -side left -expand yes -fill x
    makeNormalVector $w.up "Up Vector" up,$view $c
    pack $w.up -side left -expand yes -fill x
    global fov,$view
    frame $w.f -relief groove -borderwidth 2
    pack $w.f
    scale $w.f.fov -orient horizontal -variable fov,$view \
	    -from 0 -to 180 -label "Field of View:" \
	    -showvalue true -tickinterval 30 \
	    -digits 3 \
	    -command $c
    pack $w.f.fov -expand yes -fill x
}

proc makeBackgroundPopup {rid} {
    set w .bg$rid
    toplevel $w
    wm title $w "Background"
    wm iconname $w background
    wm minsize $w 100 100
    set c "$rid redraw "
    makeColorPicker $w bgcolor,$rid $c ""
}

proc updateMode {rid msg} {
}

proc addObject {rid objid name} {
    set w .$rid
    set m $w.mframe.f
    checkbutton $m.objlist.canvas.frame.obj$objid -text $name \
	    -relief flat -variable "$name,$rid" -command "$rid redraw"
    pack $m.objlist.canvas.frame.obj$objid -side top -anchor w
}

proc addObject2 {rid objid} {
    set w .$rid
    set m $w.mframe.f
    pack $m.objlist.canvas.frame.obj$objid -side top -anchor w
}

proc removeObject {rid objid} {
    set w .$rid
    set m $w.mframe.f
    pack forget $m.objlist.canvas.frame.obj$objid
}
