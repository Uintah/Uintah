
catch {rename TextureVolVis ""}

itcl_class Kurt_VolVis_TextureVolVis {
    inherit Module
    constructor {config} {
	set name TextureVolVis
	set_defaults
    }
    method set_defaults {} {
	global $this-draw_mode
	set $this-draw_mode 0
	global $this-num_slices
	set $this-num_slices 32
	global $this-alpha_scale
	set $this-alpha_scale 0.075
	global $this-render_style
	set $this-render_style 0
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
	label $w.f.l -text "Rendering Style"
	radiobutton $w.f.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n

	radiobutton $w.f.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n

	radiobutton $w.f.modea -text "Attenuate" -relief flat \
		-variable $this-render_style -value 2 \
		-anchor w -command $n

	pack $w.f.l $w.f.modeo $w.f.modem $w.f.modea \
		-side top -fill x

	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -padx 2 -pady 2 -fill x
	
	label $w.f2.l -text "View Mode"
	radiobutton $w.f2.full -text "Full Resolution" -relief flat \
		-variable $this-draw_mode -value 0 \
		-anchor w -command $n

	radiobutton $w.f2.los -text "Line of Sight" -relief flat \
		-variable $this-draw_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.f2.roi -text "Region of Influence" -relief flat \
		-variable $this-draw_mode -value 2 \
		-anchor w -command $n

	pack $w.f2.l $w.f2.full $w.f2.los $w.f2.roi \
		-side top -fill x




	global $this-num_slices
	scale $w.nslice -variable $this-num_slices \
		-from 4 -to 1024 -label "Number of Slices" \
		-showvalue true \
		-orient horizontal \


	global $this-alpha_scale
	scale $w.stransp -variable $this-alpha_scale \
		-from 0.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.000001\
		-orient horizontal \

	pack $w.stransp $w.nslice  -side top -fill x

	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x
	bind $w.nslice <ButtonRelease> $n
	bind $w.stransp <ButtonRelease> $n
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x
    }
}
