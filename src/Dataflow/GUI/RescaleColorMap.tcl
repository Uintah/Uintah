itcl_class PSECommon_Visualization_RescaleColorMap { 
    inherit Module 

    protected  bVar
    constructor {config} { 
		puts "constructing RescaleColorMap"
        set name RescaleColorMap 
        set_defaults 
    } 
  
    method set_defaults {} { 
	global $this-isFixed
	global $this-min
	global $this-max
	set bVar 0
	set $this-isFixed 0
	set $this-min 0
	set $this-max 1
    }   

    method ui {} { 
	global $this-isFixed
	global $this-min
	global $this-max

	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    wm deiconify $w
	    raise $w 
	    return; 
	} 
	
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
	entry $w.f3.e1 -textvariable $this-min

	label $w.f3.l2 -text "max:  "
	entry $w.f3.e2 -textvariable $this-max
	pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
	    -expand yes -fill x -padx 2 -pady 2

	bind $w.f3.e1 <Return> "$this-c needexecute"
	bind $w.f3.e2 <Return> "$this-c needexecute"

	button $w.close -text Close -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x

	if { [set $this-isFixed] } {
	    $w.f2.b select
	    $this fixedScale
	} else {
	    $w.f1.b select
	    $this autoScale
	}
    }

    method autoScale { } {
	global $this-isFixed
	set w .ui[modname]
	
	set $this-isFixed 0

	set color "#505050"

	$w.f3.l1 configure -foreground $color
	$w.f3.e1 configure -state disabled -foreground $color
	$w.f3.l2 configure -foreground $color
	$w.f3.e2 configure -state disabled -foreground $color


	$this-c needexecute
    }

    method fixedScale { } {
	global $this-isFixed
	set w .ui[modname]

	set $this-isFixed 1


	$w.f3.l1 configure -foreground black
	$w.f3.e1 configure -state normal -foreground black
	$w.f3.l2 configure -foreground black
	$w.f3.e2 configure -state normal -foreground black
	
    }
}
