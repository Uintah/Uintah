#
#

itcl_class ImageTest {
    inherit Module
    constructor {config} {
	set name ImageTest
	set_defaults
    }
    method set_defaults {} {
	global $this-scale
	set $this-scale 1.0
	global $this-offset
	set $this-offset 0
#	$this-c needexecute

    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20
	frame $w.f -width 30
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	#expscale $w.scale -orient horizontal -label "Scale:" 
	#	-variable $this-scale -command $n
	#$w.scale-win- configure
	#pack $w.scale -fill x -pady 2
	#expscale $w.offset -orient horizontal -label "Offset:" 
#		-variable $this-offset -command $n
#	pack $w.offset -fill x -pady 2
    }
}
