
catch {rename DaveW_EGI_Anneal ""}

itcl_class DaveW_EGI_Anneal {
    inherit Module
    constructor {config} {
	set name Anneal
	set_defaults
    }
    method set_defaults {} {
	global $this-angle
	global $this-sb
	global $this-ss
	global $this-st
	global $this-sbout
	global $this-ssout
	global $this-stout
	global $this-err
	global $this-niter0
	global $this-niter
	global $this-cool
	global $this-ratio
	set $this-angle 45
	set $this-sb 0.0045
	set $this-ss 0.0000563
	set $this-st 0.0045
	set $this-niter0 200
	set $this-niter 100
	set $this-cool 0.9
	set $this-ratio 0.01
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"

#	 probably want to redeclare all your globals here
	global $this-angle
	global $this-sb
	global $this-ss
	global $this-st
	global $this-sbout
	global $this-ssout
	global $this-stout
	global $this-err
	global $this-niter0
	global $this-niter
	global $this-cool
	global $this-ratio



##### input frame 

frame $w.f.f11 -relief raised -borderwidth 2 
pack $w.f.f11 -side top -fill x

# make radiobuttons for choosing number of electrodes

frame $w.f.f11.f1 
pack $w.f.f11.f1 -side top

radiobutton $w.f.f11.f1.b19 -text "19" 
radiobutton $w.f.f11.f1.b129 -text "129"  
label $w.f.f11.f1.lab1 -text "Number of electrodes" 
pack $w.f.f11.f1.lab1 $w.f.f11.f1.b19 $w.f.f11.f1.b129 -side left -padx 10

$w.f.f11.f1.b19 select

# make slider for angle between electrodes

frame $w.f.f11.f2 
pack $w.f.f11.f2 -side top -pady 10

scale $w.f.f11.f2.s180 -from 1 -to 180 -showvalue true -orient horizontal -variable $this-angle
label $w.f.f11.f2.lab2 -text "Angle between electrodes (degrees)" 
pack $w.f.f11.f2.lab2 $w.f.f11.f2.s180 -side left -padx 10

# input conductivities 

frame $w.f.f11.f3 
pack $w.f.f11.f3

frame $w.f.f11.f3.g1 
entry $w.f.f11.f3.g1.w1 -relief sunken -width 12 -textvariable $this-sb
label $w.f.f11.f3.g1.labg1 -text "Brain" 
pack $w.f.f11.f3.g1.labg1 $w.f.f11.f3.g1.w1 -side top

frame $w.f.f11.f3.g2 
entry $w.f.f11.f3.g2.w2 -relief sunken -width 12 -textvariable $this-ss
label $w.f.f11.f3.g2.labg2 -text "Skull" 
pack $w.f.f11.f3.g2.labg2 $w.f.f11.f3.g2.w2 -side top

frame $w.f.f11.f3.g3 
entry $w.f.f11.f3.g3.w3 -relief sunken -width 12 -textvariable $this-st
label $w.f.f11.f3.g3.labg3 -text "Scalp" 
pack $w.f.f11.f3.g3.labg3 $w.f.f11.f3.g3.w3 -side top

label $w.f.f11.f3.lab3 -text "Conductivities (1/Ohm*cm)" 
pack $w.f.f11.f3.lab3 $w.f.f11.f3.g1 $w.f.f11.f3.g2  $w.f.f11.f3.g3 -side left -padx 5

# annealing interations

frame $w.f.f11.f4 
pack $w.f.f11.f4 -pady 5

frame $w.f.f11.f4.g1 
entry $w.f.f11.f4.g1.w1 -relief sunken -width 5 -textvariable $this-niter0
label $w.f.f11.f4.g1.lab1 -text "Niter0" 
pack $w.f.f11.f4.g1.lab1 $w.f.f11.f4.g1.w1 -side top

frame $w.f.f11.f4.g2 
entry $w.f.f11.f4.g2.w2 -relief sunken -width 5 -textvariable $this-niter
label $w.f.f11.f4.g2.lab2 -text "Niter" 
pack $w.f.f11.f4.g2.lab2 $w.f.f11.f4.g2.w2 -side top

label $w.f.f11.f4.lab4 -text "Number of annealing iterations" 
pack $w.f.f11.f4.lab4 $w.f.f11.f4.g1 $w.f.f11.f4.g2 -side left -padx 20

# cooling rate and termination

frame $w.f.f11.f5 
pack $w.f.f11.f5 -side top -pady 10

scale $w.f.f11.f5.s1 -from 0 -to 1 -showvalue true -orient horizontal -resolution 0.01 -variable $this-cool
label $w.f.f11.f5.lab1 -text "Cooling rate" 

scale $w.f.f11.f5.s2 -from 0 -to 10 -showvalue true -orient horizontal -variable $this-ratio
label $w.f.f11.f5.lab2 -text "Log(Tf/Ti)" 

pack $w.f.f11.f5.lab1 $w.f.f11.f5.s1 $w.f.f11.f5.lab2 $w.f.f11.f5.s2 -side left -padx 5

##### output frame

frame $w.f.f22 -relief raised -borderwidth 2 
pack $w.f.f22 -side top

# err

frame $w.f.f22.f3 
pack $w.f.f22.f3

entry $w.f.f22.f3.w1 -width 12 -textvariable $this-err
label $w.f.f22.f3.lab1 -text "RMS err (Volts)" 
pack $w.f.f22.f3.lab1 $w.f.f22.f3.w1 -side left

# annealing interations

frame $w.f.f22.f4
pack $w.f.f22.f4 -pady 5

frame $w.f.f22.f4.g1
entry $w.f.f22.f4.g1.w1 -width 12 -textvariable $this-sbout
label $w.f.f22.f4.g1.lab1 -text "Brain"  
pack $w.f.f22.f4.g1.lab1 $w.f.f22.f4.g1.w1 -side top

frame $w.f.f22.f4.g2
entry $w.f.f22.f4.g2.w2 -width 12 -textvariable $this-ssout
label $w.f.f22.f4.g2.lab2 -text "Skull"  
pack $w.f.f22.f4.g2.lab2 $w.f.f22.f4.g2.w2 -side top

frame $w.f.f22.f4.g3
entry $w.f.f22.f4.g3.w2 -width 12 -textvariable $this-stout
label $w.f.f22.f4.g3.lab2 -text "Scalp"  
pack $w.f.f22.f4.g3.lab2 $w.f.f22.f4.g3.w2 -side top

label $w.f.f22.f4.lab4 -text "Conductivities (1/Ohm*cm)"
pack $w.f.f22.f4.lab4 $w.f.f22.f4.g1 $w.f.f22.f4.g2  $w.f.f22.f4.g3 -side left -padx 5


}
}
