
#
#  OpenGLWindow
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   July 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl_class OpenGLWindow {
    constructor {config} {
	set name OpenGLWindow
	set_defaults
    }

    method set_defaults {} {
	set $this-window 0
    }

    method ui { parent w } {
	global $this-opengl-win

	set $this-opengl-win $w.gl

	opengl $w.gl -geometry 500x300 -doublebuffer true \
	    -direct true -rgba true 
	

	bind $w.gl <Map> "$parent-c map $w.gl"
	bind $w.gl <Expose> "$parent-c redraw $w.gl"
	pack $w.gl
    }

    method change_params { n } {
    }
	
}
	
	