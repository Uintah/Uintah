
#
#  BB.tcl
#
#  Written by:
#   Aleksandra Kuswik
#   Department of Computer Science
#   University of Utah
#   May 1997
#
#  Copyright (C) 1997 SCI Group
#

#
#
#
################################################################
#
#
#
################################################################


itcl_class BB {
    
    inherit Module


    #
    #
    #
    ################################################################
    #
    # construct the BB class.  called when BB is instantiated.
    #
    ################################################################
    
    constructor {config} {
	puts "WELCOME TO THE CONSTRUCTOR OF BB"
	set name BB
	set_defaults
    }
    
    
    #
    #
    #
    ################################################################
    #
    # initialize variables.
    #
    ################################################################
    
    method set_defaults {} {

	# set protected variables and globals
	
	set redrawing           0
	set $this-eview       100
	set $this-raster      100

	set changing            0
    }
    
    #
    #
    #
    ################################################################
    #
    # raise the GL window or create it if not yet created.
    #
    ################################################################

    method raiseGL {} {
	
	set w .ui$this
	
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {

	    # initialize geometry and placement of the widget
	    
	    toplevel $w.gl
	    wm geometry $w.gl =600x600+300-200

	    # create an OpenGL widget
	    
	    opengl $w.gl.gl -geometry 600x600 -doublebuffer true \
		    -direct true -rgba true -redsize 2 -greensize 2 \
		    -bluesize 2 -depthsize 0

	    # every time the OpenGL widget is displayed, redraw it
	    
	    bind $w.gl.gl <Expose> "$this redraw_when_idle"
	    
	    # place the widget on the screen

	    pack $w.gl.gl -fill both -expand 1
	}
    }
    
    
    #
    #
    #
    ################################################################
    #
    # this method defines what happens when the UI button is
    # clicked on the module.  if already existant, it is raised,
    # otherwise, it is created.
    #
    ################################################################

    method ui {} {
	set w .ui$this
	
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	
	toplevel $w
	frame $w.f

	
	# create a button for each function
	
	frame $w.f.viewstuff
	$this makeViewPopup
	frame $w.f.rastersize
	$this adjustRasterSize
	
	# place the buttons in a window

        pack $w.f.viewstuff $w.f.rastersize                  \
		-expand yes -fill x -pady 2 -padx 2
	pack $w.f

	# raise the OpenGL display window
	
	raiseGL
    }
    


    #
    #
    #
    ################################################################
    #
    # allows the user to alter the view.
    #
    ################################################################

    method makeViewPopup {} {
	
	set w .view$this
	
	if {[winfo exists $w]} {
	    raise $w
	} else {
	    
	    # initialize variables
	    
	    toplevel $w
	    wm title $w "View"
	    wm iconname $w view

	    set view $this-eview
	    
	    # allow to adjust the eye and look at point, as well
	    # as the normal vector and the field of view angle.
	    
	    makePoint $w.eyep "Eye Point" $view-eyep \
		    "$this notify_of_change_when_idle" 6c
	    makePoint $w.lookat "Look at Point" $view-lookat \
		    "$this notify_of_change_when_idle" "6c"
	    makeNormalVector $w.up "Up Vector" $view-up \
		    "$this notify_of_change_when_idle"
	    
	    # place the points in a window
	    
	    pack $w.eyep -side left -expand yes -fill x
	    pack $w.lookat -side left -expand yes -fill x
	    pack $w.up -side left -expand yes -fill x
	}
    }


    #
    #
    #
    ################################################################
    #
    # allows the user to adjust the raster size.
    #
    ################################################################

    method adjustRasterSize {} {

	set w .adjustRS$this

	if {[winfo exists $w]} {
	    raise $w
	} else {

	    # initialize variables
	    
	    toplevel $w
	    wm title $w "Raster Size"

	    # create the scale

	    frame $w.f -relief groove -borderwidth 2
	    scale $w.f.res -orient horizontal -variable $this-raster \
		    -from 100 -to 600 -label "Raster Size:" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 12c  \
		    -command "$this notify_of_change_when_idle"
	    
	    # place the scales in a window
	    
	    pack $w.f.res -expand yes -fill x
	    pack $w.f

	}
    }


    #
    #
    #
    ################################################################
    #
    # unsets the flag, calls a redraw
    #
    ################################################################

    method redraw {} {
	set redrawing 0
	$this-c redraw_all
    }


    #
    #
    #
    ################################################################
    #
    # if a few redraws have been queued, and one has just taken
    # place, do not redraw a second time.
    #
    ################################################################

    method redraw_when_idle {} {

	if { ! $redrawing } {
	    after idle $this redraw
	    set redrawing 1
	}
    }

    method change {} {
	set changing 0
	$this-c view_changed
    }

    method notify_of_change_when_idle {joy} {

	if { ! $changing } {
	    after idle $this change
	    set changing 1
	}
    }
    protected redrawing
    protected changing
}
