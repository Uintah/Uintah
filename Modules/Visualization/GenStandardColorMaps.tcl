itcl_class GenStandardColorMaps { 

    inherit Module 
    protected exposed

    constructor {config} { 
        set name GenStandardColorMaps 
        set_defaults 
    } 
  


    method set_defaults {} { 
        global $this-tcl_status 
	set exposed 0
    } 
  

    method ui {} { 
	global $this-minRes
	global $this-resolution
	global $this-mapType

        set w .ui$this 
        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
            return; 
        } 

	set type ""
  
        toplevel $w 
        wm minsize $w 200 50 
  
        #set n "$this-c needexecute " 
	set n "$this change"

	frame $w.f -relief flat -borderwidth 2
	pack $w.f -side top -expand yes -fill x 
	button $w.f.b -text "close" -command "$this close"
	pack $w.f.b -side left -expand yes -fill y

        frame $w.f.f1 -relief sunken -height 25 -borderwidth 2 
	pack $w.f.f1 -side right -padx 2 -pady 2 -expand yes -fill x

	canvas $w.f.f1.canvas -bg "#ffffff" -height 25
	pack $w.f.f1.canvas -anchor w -expand yes -fill x

	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -side left -padx 2 -pady 2 -expand yes -fill both
	
	make_labeled_radio $w.f2.types "ColorMaps" $n  top \
	    $this-mapType { {"Gray" 0} {"Inverse Gray" 1 } 
		{"Rainbow" 2} {"Darkhue" 3} {"Blackbody" 4 }}

	pack $w.f2.types -in $w.f2 -side left
	frame $w.f2.f3 -relief groove -borderwidth 2	
	pack $w.f2.f3 -side left -padx 2 -pady 2 -expand yes -fill both
	label $w.f2.f3.label -text "Resolution"
	pack $w.f2.f3.label -side top -pady 2
	scale $w.f2.f3.s -from [set $this-minRes] -to 255 -state normal \
	    -orient horizontal  -variable $this-resolution 
	pack $w.f2.f3.s -side top -padx 2 -pady 2 -expand yes -fill x

	bind $w.f2.f3.s <ButtonRelease> "$this update"
	bind $w.f.f1.canvas <Expose> "$this canvasExpose"
	$this update

    }

    method change {} {
	global $this-minRes
	global $this-resolution
	global $this-mapType
	set w .ui$this
	switch  [set $this-mapType] {
	    0  -
	    1  { set $this-minRes 2}
	    2  { set $this-minRes 12}
	    3  { set $this-minRes 19}
	    4  { set $this-minRes 13}
            default {set $this-minRes 19}
	}
	$w.f2.f3.s configure -from [set $this-minRes]

	$this update
    }

    method update { } {
	$this redraw
	$this-c needexecute
    }

    method close {} {
	set w .ui$this
	set exposed 0
	destroy $w
    }
    method canvasExpose {} {
	set w .ui$this

	if { [winfo viewable $w.f.f1.canvas] } { 
	    if { $exposed } {
		return
	    } else {
		set exposed 1
		$this redraw
	    } 
	} else {
	    return
	}
    }
	
    method redraw {} {
	global $this-resolution
	set w .ui$this

	set w .ui$this
	set n [set $this-resolution]
	set canvas $w.f.f1.canvas
	$canvas delete all
	set cw [winfo width $canvas]
	set ch [winfo height $canvas]
	set dx [expr $cw/double($n)] 
	set x 0
	set colors [$this-c getcolors]
	for {set i 0} {$i < $n} {incr i 1} {
	    set color [lindex $colors $i]
	    set oldx $x
	    set x [expr ($i+1)*$dx]
	    $canvas create rectangle \
		$oldx 0 $x $ch -fill $color -outline $color
	}
    }	
}
