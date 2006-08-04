
#  ViewTensors.tcl

package require Iwidgets 3.0

itcl_class Yarden_Visualization_ViewTensors {
    
    inherit Module
    
    constructor {config} {
	set name ViewTensors
	set_defaults
    }
    
    method set_defaults {} {
	global $this-redrawing
	global $this-num-slices
	global $this-slice
	global $this-mode
	global $this-nx
	global $this-ny
	global $this-nz

	set $this-redrawing 0
	set $this-num-slices 100
	set $this-slice 0
	set $this-mode 2
    }
    

    method raiseGL {} {
	set w .ui[modname]
	
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {

	    toplevel $w.gl 
	    
	    opengl $w.gl.gl -geometry 512x700 -doublebuffer true \
		-direct true -rgba true \
		-redsize 1 -greensize 1 -bluesize 1 -depthsize 2

	    # every time the OpenGL widget is displayed, redraw it
 	    bind $w.gl.gl <Expose> "$this redraw_when_idle %w %h"
 	    bind $w.gl.gl <Configure> "$this-c configure %w %h"
	    
	    pack $w.gl.gl -fill both -expand 1
	}
    }
    
    

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		   
	    return;
	}
	
	toplevel $w
	
	iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info:"
 	set info [$w.info childsite]
	label $info.size -text "no data"
	pack $info.size -anchor w
	
	iwidgets::optionmenu $w.mode -labeltext "Slice Direction" \
	    -labelpos w  -command "$this change-mode"

	$w.mode insert end Z Y X

	iwidgets::Labeledframe $w.slice -labelpos nw -labeltext "Slice"
 	set slice [$w.slice childsite] 

	scale $slice.val -label "Slice" \
	    -variable $this-slice \
	    -from 0 -to [set $this-num-slices] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -resolution 1 \
	    -digits 4 \
	    -command "$this set-slice"

	trace variable $this-num-slices w "$this change-num-slices"

	pack $slice.val -side left -fill x
  	pack $w.info $w.mode $w.slice -side top -anchor w

	$this raiseGL
    }

    method new-info { x y z } {
	global $this-nx
	global $this-ny
	global $this-nz

	set $this-nx $x
	set $this-ny $y
	set $this-nz $z

	set w .ui[modname]
 	set info [$w.info childsite]
	$info.size configure -text "$x x $y x $z"
# 	$this change-mode
    }

    method set-slice { n } {
	global $this-slice
	
	$this-c slice [set $this-slice]
    }

    method change-num-slices {n1 n2 op} {
 	set slice [.ui[modname].slice childsite].val
	global $slice
	global $this-num-slices
	
	$slice configure -to [set $this-num-slices]
    }


    method change-mode {} {
	set w .ui[modname]
	global $this-mode
	global $this-nx
	global $this-ny
	global $this-nz
	
 	set slice [$w.slice childsite] 
	switch [$w.mode get] \
	    "Z" {  set $this-mode 2
		   set slice [.ui[modname].slice childsite] 
	  	   $slice.val configure -to [expr [set $this-nz] - 1] 
	        } \
	    "Y" {  set $this-mode 1
		   set slice [.ui[modname].slice childsite] 
		   $slice.val configure -to [expr [set $this-ny] - 1] 
                } \
	    "X" {  set $this-mode 0
		   set slice [.ui[modname].slice childsite] 
		   $slice.val  configure -to [expr [set $this-nx] -1]
                } 

	$this redraw_when_idle 0 0
    }

    method redraw { } {
 	set w .ui[modname]
	global $this-redrawing
	set $this-redrawing 0
	$this-c redraw_all [lindex [$w.gl.gl configure -geometry] 4]
    }

    method redraw_when_idle { w h } {
	global $this-redrawing
	if { ! [set $this-redrawing] } {
	    after idle $this redraw 
	    set $this-redrawing 1
	}
    }

}

