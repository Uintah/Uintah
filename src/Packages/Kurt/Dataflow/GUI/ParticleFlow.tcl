itcl_class Kurt_Visualization_ParticleFlow {
    inherit Module
    constructor {config} {
        set name ParticleFlow
        set_defaults
    }

    method set_defaults {} {
	global $this-animate
        global $this-time
        set $this-animate 0
        set $this-time 0.002
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
	frame $w.f1 
        frame $w.f2
        
        pack $w.f -side top
	pack $w.f1 $w.f2 -in $w.f -padx 2 -pady 2 -fill x -side left
	
	set n "$this-c needexecute"

        checkbutton $w.f1.animate -text Animate -variable $this-animate
        label $w.f2.l -text "Particle life time increment"
        scale $w.f2.scale -to 0.01 -from 0.0002 -orient horizontal \
            -variable $this-time -resolution 0.0002
        pack  $w.f1.animate 
        pack $w.f2.l $w.f2.scale  -side top


        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


