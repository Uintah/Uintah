
itcl_class GenTransferFunc {
    inherit Module
    constructor {config} {
	set name GenTransferFunc
	set_defaults
    }
    method set_defaults {} {
	global $this-rgbhsv
	set $this-rgbhsv 1
#	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    doGL
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"
	
	global $this-rgbhsv
	global $this-linespline

	set $this-rgbhsv 0
	set $this-linespline 0
	
	doGL

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
            
            opengl $w.f.gl1.gl -geometry 512x256 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2  -depthsize 0

            # every time the OpenGL widget is displayed, redraw it
            
            bind $w.f.gl1.gl <Expose> "$this-c expose 0"
            bind $w.f.gl1.gl <ButtonPress> "$this-c mouse 0 down %x %y %b"
            bind $w.f.gl1.gl <ButtonRelease> "$this-c mouse 0 release %x %y %b"
	    bind $w.f.gl1.gl <Motion>        "$this-c mouse 0 motion %x %y"

            # place the widget on the screen

            pack $w.f.gl1.gl -fill both -expand 1
        }

        if {[winfo exists $w.f.gl2]} {
            raise $w
        } else {

            # initialize geometry and placement of the widget
            
	    frame $w.f.gl2
	    pack $w.f.gl2 -padx 2 -pady 2

            # create an OpenGL widget
            
            opengl $w.f.gl2.gl -geometry 512x256 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0

            # every time the OpenGL widget is displayed, redraw it
            
            bind $w.f.gl2.gl <Expose> "$this-c expose 1"
            bind $w.f.gl2.gl <ButtonPress> "$this-c mouse 1 down %x %y %b"
            bind $w.f.gl2.gl <ButtonRelease> "$this-c mouse 1 release %x %y %b"
	    bind $w.f.gl2.gl <Motion>        "$this-c mouse 1 motion %x %y"

            # place the widget on the screen

            pack $w.f.gl2.gl -fill both -expand 1
        }

        if {[winfo exists $w.f.gl3]} {
            raise $w
        } else {

            # initialize geometry and placement of the widget
            
	    frame $w.f.gl3
	    pack $w.f.gl3 -padx 2 -pady 2

            # create an OpenGL widget

# Use this one for machines without alpha            
            opengl $w.f.gl3.gl -geometry 512x64 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0

# Use this one for machines with alpha
#            opengl $w.f.gl3.gl -geometry 512x64 -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -alphasize 2 -depthsize 0

            # every time the OpenGL widget is displayed, redraw it
            
            bind $w.f.gl3.gl <Expose> "$this-c expose 2"

            # place the widget on the screen

            pack $w.f.gl3.gl -fill both -expand 1
        }
        
    }

}
