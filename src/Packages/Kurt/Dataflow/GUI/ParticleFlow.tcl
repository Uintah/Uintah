itcl_class Kurt_Visualization_ParticleFlow {
    inherit Module
    constructor {config} {
        set name ParticleFlow
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        wm geometry $w ""

        frame $w.f
	frame $w.f1
        frame $w.f2 -borderwidth 2 -relief groove 
        frame $w.f3 -borderwidth 2  
        frame $w.f4 -borderwidth 2 
        frame $w.f5 -borderwidth 2 -relief groove
       
#        pack $w.f -side top -expand yes -fill both
	pack $w.f2 $w.f3 $w.f4 $w.f5 $w.f1  -padx 2 -pady 2 \
            -expand yes -side top -anchor w -fill x -padx 2 -pady 2
	
	set n "$this-c needexecute"

        checkbutton $w.f1.animate -text Animate -variable $this-animate 
        checkbutton $w.f1.freeze -text "Freeze Particles" \
            -variable $this-freeze-particles -command $n
        label $w.f2.l -text "Particle life time decrement"
        scale $w.f2.scale -to 1.0 -from 0.0002 -orient horizontal \
            -variable $this-time -resolution 0.0002 
        pack  $w.f1.animate $w.f1.freeze -anchor nw
        pack $w.f2.l $w.f2.scale  -side top -expand yes -fill x

        frame $w.f3.f1
        frame $w.f3.f2
        pack  $w.f3.f1 $w.f3.f2 -side left -expand yes -fill x -anchor w \
            -padx 2 -pady 2

        label $w.f3.f1.l -text "iterations: "
        entry $w.f3.f2.e -textvariable "$this-nsteps"
        pack $w.f3.f1.l $w.f3.f2.e

        frame $w.f4.f1
        frame $w.f4.f2
        pack  $w.f4.f1 $w.f4.f2 -side left -expand yes -fill x -anchor w \
            -padx 2 -pady 2

        label $w.f4.f1.l -text "step size: "
        scale $w.f4.f2.s -variable "$this-step-size" -from 0.00001 -to 1.0 \
            -resolution 0.00002 -orient horizontal
        pack $w.f4.f1.l $w.f4.f2.s -expand yes -fill x

        frame $w.f5.f1
        frame $w.f5.f2
        pack  $w.f5.f1 $w.f5.f2 -side left -expand yes -fill x -anchor w \
            -padx 2 -pady 2

        label $w.f5.f1.l -text "2\u207f x 2\u207f  particles.  n = "
        scale $w.f5.f2.s -variable "$this-nparticles" -from 1 -to 20 \
            -orient horizontal
        pack $w.f5.f1.l $w.f5.f2.s -expand yes -fill x -anchor s

        button $w.recompute -text "Recompute Points from start points" \
            -command "set $this-recompute-points 1; $n"
        button $w.reset -text "Reset Frame Widget" \
            -command "set $this-widget-reset 1; $n"

        pack $w.recompute $w.reset -side top -expand yes -fill x

        bind $w.f3.f2.e <Return> $n 
        bind $w.f4.f2.s <ButtonRelease> $n 
        bind $w.f5.f2.s <ButtonRelease> $n 

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


