
itcl_class SCIRun_Image_Gauss {
    inherit Module
    constructor {config} {
	set name Gauss
	set_defaults
    }
    method set_defaults {} {
#	$this-c needexecute
    	global $this-sigma
	set $this-sigma 1.0
    	global $this-size
	set $this-size 3
	global $this-hardware
	set $this-hardware 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    doGL
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
#	set n "$this-c needexecute"
	
	set n "$this-c needexecute "

        label $w.f.lab -text "Sigma: "
	entry $w.f.n1 -relief sunken -width 7 -textvariable $this-sigma
	label $w.f.lab2 -text " Kernel Size:"
	entry $w.f.n2 -relief sunken -width 7 -textvariable $this-size
	pack $w.f.lab $w.f.n1 $w.f.lab2 $w.f.n2 -side left

	checkbutton $w.f.v1 -text "Use Hardware Convolution" -relief flat \
		-variable $this-hardware
	pack $w.f.v1 -side bottom
	

        button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit -side bottom

#	doGL

    }

    method rflush {} {
	$this-c needexecute
    }
    method doGL {} {
        
        set w .ui[modname]
        
        if {[winfo exists $w.f.gl1]} {
            raise $w
        } else {

            # initialize geometry and placement of the widget
            
	    frame $w.f.gl1
	    pack $w.f.gl1 -padx 2 -pady 2

            # create an OpenGL widget
            
#            opengl $w.f.gl1.gl -geometry 800x600 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2  -depthsize 0

            # every time the OpenGL widget is displayed, redraw it
            
#            bind $w.f.gl1.gl <Expose> "$this-c expose 0"
            bind $w.f.gl1.gl <ButtonPress> "$this-c mouse 0 down %x %y %b"
            bind $w.f.gl1.gl <ButtonRelease> "$this-c mouse 0 release %x %y %b"
	    bind $w.f.gl1.gl <Motion>        "$this-c mouse 0 motion %x %y %b"

            # place the widget on the screen

            pack $w.f.gl1.gl -fill both -expand 1
        }

    }

}
