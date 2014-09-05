
catch {rename IsoSurfaceMRSG ""}

itcl_class IsoSurfaceMRSG {
    inherit Module
    constructor {config} {
	set name IsoSurfaceMRSG
	set_defaults
    }
    method set_defaults {} {
	global $this-isoval
	global $this-min $this-max
	global $this-fmin $this-fmax
	global $this-nframe
	global $this-dointerp 

	set $this-fmin 0
	set $this-fmax 1.0
	set $this-nframe 10
	set $this-dointerp 0
	set $this-nsamp 10

	set $this-min 0
	set $this-max 100
	puts "set_defaults"
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	global $this-min $this-max $this-isoval
	scale $w.f.isoval -variable $this-isoval \
		-from [set $this-min] -to [set $this-max] -label "IsoValue:" \
		-resolution 0.000001 -showvalue true \
		-orient horizontal \
		-command $n -state normal
	pack $w.f.isoval -side top -fill x
	scale $w.f.nframes -variable $this-nframe \
		-from 1 -to 300 -label "Number of Frames:" \
		-showvalue true \
		-orient horizontal \
		-command $n -state normal
	pack $w.f.nframes -side top -fill x
	scale $w.f.fmin -variable $this-fmin \
		-from 0 -to 1.0 -label "Start Frame"\
		-resolution 0.001 \
		-orient horizontal \
		-command $n -state normal

	scale $w.f.fmax -variable $this-fmax \
		-from 0 -to 1.0 -label "End Frame"\
		-orient horizontal \
		-resolution 0.001 \
		-command $n -state normal

	pack $w.f.fmin $w.f.fmax -side top -fill x

	checkbutton $w.f.dointerp -text "Interpolate Time" -variable $this-dointerp
	pack $w.f.dointerp -side top -fill x
		
    }
    method set_minmax {min max} {
	set w .ui[modname]
	global $this-min $this-max
	set $this-min $min
	set $this-max $max
	$w.f.isoval configure -from $min -to $max
    }
}
