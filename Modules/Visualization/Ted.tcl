
itcl_class Ted {
    inherit Module
    constructor {config} {
	set name Ted
	set_defaults
    }
    method set_defaults {} {
#	$this-c needexecute
    	global $this-zoom
	set $this-zoom 1
    	global $this-normal
	set $this-normal 0
    	global $this-negative
	set $this-negative 1
        global $this-x
        set $this-x 0
        global $this-y
        set $this-y 0

    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    doGL
	    jrefresh
	    resetc
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	
	set n "$this-c needexecute "

	expscale $w.zoom -orient horizontal -label "zoom:" -variable $this-zoom -command "$this jrefresh"
	$w.zoom-win- configure
	pack $w.zoom -fill x -pady 2

        checkbutton $w.f.v1 -text "Normalize" -relief flat \
		-variable $this-normal  
        checkbutton $w.f.v2 -text "Show Negatives" -relief flat \
		-variable $this-negative  

        button $w.f.reset -text " Reset RasterPos " -command "$this-c resetraster"
        button $w.f.refresh -text " Refresh " -command "$this jrefresh2"

	doGL

	pack $w.f.v1 $w.f.v2 $w.f.reset $w.f.refresh -side left

	frame $w.f.r4
	pack $w.f.r4
	
	label $w.f.r4.lab -text "X: "
	entry $w.f.r4.n1 -relief sunken -width 7 -textvariable $this-x
	label $w.f.r4.lab2 -text " Y: "
	entry $w.f.r4.n2 -relief sunken -width 7 -textvariable $this-y

	button $w.f.r4.doit -text " View Grid Coord " -command "$this viewg"

	pack $w.f.r4.lab $w.f.r4.n1 $w.f.r4.lab2 $w.f.r4.n2 $w.f.r4.doit -side left

    }
    method viewg {} {
	global $this-x
	global $this-y

	$this-c viewgrid [set $this-x] [set $this-y]
    }
    method resetc {} {
	$this-c resetraster 0
#	$this-c needexecute
    }
    method jrefresh {zval} {
	global $this-zoom

	$this-c justrefresh [set $this-zoom]
    }
    method jrefresh2 {} {
	global $this-zoom

	$this-c needexecute
    }
    method doGL {} {
        
        set w .ui$this
        
        if {[winfo exists $w.f.gl1]} {
            raise $w
        } else {

            # initialize geometry and placement of the widget
            
	    frame $w.f.gl1
	    pack $w.f.gl1 -padx 2 -pady 2

            # create an OpenGL widget
            
            opengl $w.f.gl1.gl -geometry 800x600 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2  -depthsize 0

            # every time the OpenGL widget is displayed, redraw it
            
            bind $w.f.gl1.gl <Expose> "$this-c expose 0"
            bind $w.f.gl1.gl <ButtonPress> "$this-c mouse 0 down %x %y %b"
            bind $w.f.gl1.gl <ButtonRelease> "$this-c mouse 0 release %x %y %b"
	    bind $w.f.gl1.gl <Motion>        "$this-c mouse 0 motion %x %y %b"

            # place the widget on the screen

            pack $w.f.gl1.gl -fill both -expand 1
        }

    }

}
