
catch {rename GLTextureBuilder ""}

itcl_class Kurt_Vis_GLTextureBuilder {
    inherit Module
    constructor {config} {
	set name GLTextureBuilder
	set_defaults
    }
    method set_defaults {} {
	global $this-max_brick_dim
	set $this-max_brick_dim 0 
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	frame $w.f.dimframe -relief groove -border 2
	label $w.f.dimframe.l -text "Brick Size Cubed"
	pack $w.f.dimframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.dimframe.l -side top -fill x

	if { [set $this-max_brick_dim] != 0 } {
	    $this SetDims [set $this-max_brick_dim]
	}

	button $w.b -text Close -command "wm withdraw $w"
	pack $w.b -side bottom -fill x
    }


    method SetDims { val } {
	global $this-max_brick_dim
	set $this-max_brick_dim $val
	set w .ui[modname]

	set vals  [format "%i %i %i" [expr $val/4] [expr $val/2] $val] 
	set vals [split $vals]
	if {![winfo exists $w]} {
	    return
	}
	if {[winfo exists $w.f.dimframe.f]} {
	    destroy $w.f.dimframe.f
	}

	frame $w.f.dimframe.f -relief flat
	pack $w.f.dimframe.f -side top -fill x
	set f $w.f.dimframe.f
	for {set i 0} {$i < 3} { incr i} {
	    set v [lindex $vals $i]
	    radiobutton $f.brickdim$v -text $v -relief flat \
		-variable $this-max_brick_dim -value $v \
		-command "$this-c needexecute"
	    pack $f.brickdim$v -side left -padx 2 -fill x
	}
    }
}
