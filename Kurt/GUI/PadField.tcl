
catch {rename PadField ""}

itcl_class Kurt_Vis_PadField {
    inherit Module
    constructor {config} {
	set name PadField
	set_defaults
    }
    method set_defaults {} {
	global $this-pad_mode
	set $this-draw_mode 0
	global $this-xpad
	set $this-xpad 0
	global $this-ypad
	set $this-ypad 0
	global $this-zpad
	set $this-zpad 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	global $this-pad_mode
	radiobutton $w.f.modep -text "Pad by extending  X, Y, & Z" \
	    -relief flat \
	    -variable $this-pad_mode -value 0 \
	    -command $n

	radiobutton $w.f.modev -text "Pad Border (center old field)" \
	        -relief flat \
		-variable $this-pad_mode -value 1 \
		-command $n

	global $this-xpad
	scale $w.f.xpad -variable $this-xpad \
		-from 0 -to 100 -label "Padding in X" \
		-showvalue true \
		-orient horizontal 
	global $this-ypad
	scale $w.f.ypad -variable $this-ypad \
		-from 0 -to 100 -label "Padding in Y" \
		-showvalue true \
		-orient horizontal 
	global $this-zpad
	scale $w.f.zpad -variable $this-zpad \
		-from 0 -to 100 -label "Padding in Z" \
		-showvalue true \
		-orient horizontal 

	pack $w.f.modep $w.f.modev $w.f.xpad $w.f.ypad $w.f.zpad \
		-side top -fill x

	button $w.f.exec -text "Execute" -command $n
	pack $w.f.exec -side top -fill x
    }
}
