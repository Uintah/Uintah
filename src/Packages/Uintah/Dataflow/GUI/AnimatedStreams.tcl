
catch {rename AnimatedStreams ""}

itcl_class Uintah_Visualization_AnimatedStreams {
    inherit Module
    constructor {config} {
	set name AnimatedStreams
	set_defaults
    }
    method set_defaults {} {
	global $this-pause
	set $this-pause 1
	global $this-normals
	set $this-normals 0
	global $this-stepsize
	set $this-stepsize 0.1
	global $this-linewidth
	set $this-linewidth 2
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	#wm minsize $w 250 300
	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -padx 2 -pady 2 -fill x

	global $this-pause
	checkbutton $w.f2.pause -text "Pause" -relief flat \
		-variable $this-pause -onvalue 1 -offvalue 0 \
		-anchor w -command $n

	global $this-normals
	checkbutton $w.f2.normals -text "Normals On" -relief flat \
		-variable $this-normals -onvalue 1 -offvalue 0 \
		-anchor w -command $n

	pack  $w.f2.pause $w.f2.normals \
		-side top -fill x

	
	set r [expscale $w.stepsize \
		-label "Step Size:" \
		-orient horizontal \
		-variable $this-stepsize]
	pack $w.stepsize -side top -fill x

#	scale $w.steps -variable $this-stepsize \
#		-from 0 -to 1 -label "Step Size" \
#		-showvalue true \
#	        -resolution 0.01 \
#		-orient horizontal

#	pack $w.steps  -side top -fill x

	bind $w.stepsize <ButtonRelease> $n
	
	global $this-linewidth
	scale $w.linewidth -variable $this-linewidth \
	    -from 1 -to 4 -label "Stream width" \
	    -showvalue true \
	    -resolution 1.0 \
	    -orient horizontal 

	pack $w.linewidth -side top -fill x

	button $w.reset -text "Reset Streams" -command "$this-c reset_streams"
	pack $w.reset -side top -fill x

	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x
    }
}
