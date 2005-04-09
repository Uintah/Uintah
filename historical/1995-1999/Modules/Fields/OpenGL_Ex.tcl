
#
#  OpenGL_Ex.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   April 1996
#
#  Copyright (C) 1996 SCI Group
#

itcl_class OpenGL_Ex {
    inherit Module
    constructor {config} {
	set name OpenGL_Ex
	set_defaults
    }
    
    method set_defaults {} {
	$this-c needexecute
    }
    
    method raiseGL {} {
	set w .ui$this
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm geometry $w.gl =600x600+300-200
	    wm minsize $w.gl 600 600
	    wm maxsize $w.gl 600 600
	    opengl $w.gl.gl -geometry 600x600 -doublebuffer false -direct false -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0
	    bind $w.gl.gl <Expose> "$this-c redraw_all"
	    pack $w.gl.gl -fill both -expand 1
	}
    }
	
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	frame $w.f
	button $w.f.b -text "Redraw" -command "$this-c redraw_all"
	pack $w.f.b
	pack $w.f
	raiseGL
    }
}

