itcl_class Uintah_Visualization_RescaleColorMapForParticles { 

    inherit Module 

    constructor {config} { 
        set name RescaleColorMapForParticles 
        set_defaults 
    } 
  
    method set_defaults {} { 
        global $this-tcl_status 
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode
	set exposed 0
	set $this-scaleMode "auto"
    } 
  

    method ui {} { 
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode

        set w .ui[modname] 
        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
            return; 
        } 

	set type ""
  
        toplevel $w 
        wm minsize $w 200 50 
	frame $w.f1 -relief flat
	pack $w.f1 -side top -expand yes -fill x
	radiobutton $w.f1.b -text "Auto Scale"  -variable bVar -value 0 \
	    -command "$this autoScale"
	pack $w.f1.b -side left

	frame $w.f2 -relief flat
	pack $w.f2 -side top -expand yes -fill x
	radiobutton $w.f2.b -text "Fixed Scale"  -variable bVar -value 1 \
	    -command "$this fixedScale"
	pack $w.f2.b -side left

	frame $w.f3 -relief flat
	pack $w.f3 -side top -expand yes -fill x
	
	label $w.f3.l1 -text "min:  "
	entry $w.f3.e1 -textvariable $this-minVal

	label $w.f3.l2 -text "max:  "
	entry $w.f3.e2 -textvariable $this-maxVal
	pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
	    -expand yes -fill x -padx 2 -pady 2

	bind $w.f3.e1 <Return> "$this-c needexecute"
	bind $w.f3.e2 <Return> "$this-c needexecute"

	button $w.close -text Close -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x

	$this [set $this-scaleMode]Scale

    }	

    method fixedScale {} {
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode
	set w .ui[modname]

	$w.f2.b select

	$w.f3.l1 configure -foreground black
	$w.f3.e1 configure -state normal -foreground black
	$w.f3.l2 configure -foreground black
	$w.f3.e2 configure -state normal -foreground black

	set $this-scaleMode "fixed"
	#$this-c needexecute
    }

    method autoScale {} {
	global $this-scaleMode
	set w .ui[modname]

	$w.f1.b select

	set color "#505050"

	$w.f3.l1 configure -foreground $color
	$w.f3.e1 configure -state disabled -foreground $color
	$w.f3.l2 configure -foreground $color
	$w.f3.e2 configure -state disabled -foreground $color

	set $this-scaleMode "auto"
	$this-c needexecute
    }
}
