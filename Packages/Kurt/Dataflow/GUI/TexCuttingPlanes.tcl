
catch {rename TexCuttingPlanes ""}

itcl_class Kurt_Vis_TexCuttingPlanes {
    inherit Module
    constructor {config} {
	set name TexCuttingPlanes
	set_defaults
    }
    method set_defaults {} {
	global $this-drawX
	global $this-drawY
	global $this-drawZ
	global $this-drawView
	set $this-drawX 0
	set $this-drawY 0
	set $this-drawZ 0
	set $this-drawView 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	global $this-render_style
	label $w.f.l -text "Select Plane(s)"
	checkbutton $w.f.xp -text "X plane" -relief flat \
	    -variable $this-drawX -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $n"

	checkbutton $w.f.yp -text "Y plane" -relief flat \
	    -variable $this-drawY -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $n"

	checkbutton $w.f.zp -text "Z plane" -relief flat \
	    -variable $this-drawZ -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $n"

	checkbutton $w.f.vp -text "View plane" -relief flat \
	    -variable $this-drawView -onvalue 1 -offvalue 0 \
	    -anchor w -command \
	    "set $this-drawX 0; set $this-drawY 0; set $this-drawZ 0; $n"


	pack $w.f.l $w.f.xp $w.f.yp $w.f.zp $w.f.vp \
		-side top -fill x

	frame $w.f.x
	button $w.f.x.plus -text "+" -command "$this-c MoveWidget xplus; $n"
	label $w.f.x.label -text " X "
	button $w.f.x.minus -text "-" -command "$this-c MoveWidget xminus; $n"
	pack $w.f.x.plus $w.f.x.label $w.f.x.minus -side left -fill x -expand 1
	pack $w.f.x -side top -fill x -expand 1

	frame $w.f.y
	button $w.f.y.plus -text "+" -command "$this-c MoveWidget yplus; $n"
	label $w.f.y.label -text " Y "
	button $w.f.y.minus -text "-" -command "$this-c MoveWidget yminus; $n"
	pack $w.f.y.plus $w.f.y.label $w.f.y.minus -side left -fill x -expand 1
	pack $w.f.y -side top -fill x -expand 1

	frame $w.f.z
	button $w.f.z.plus -text "+" -command "$this-c MoveWidget zplus; $n"
	label $w.f.z.label -text " Z "
	button $w.f.z.minus -text "-" -command "$this-c MoveWidget zminus; $n"
	pack $w.f.z.plus $w.f.z.label $w.f.z.minus -side left -fill x -expand 1
	pack $w.f.z -side top -fill x -expand 1

	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x
    }
}
