#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

#  Diagram.tcl
#  Written by:
#   Yarden Livnat
#   Department of Computer Science
#   University of Utah
#   July 2001
#  Copyright (C) 2001 SCI Group

package require Iwidgets 3.0

itcl::class Diagram {

    variable menu
    variable tb
    variable ui
    variable ogl
    variable parent
    variable name

    variable initialized
    variable last
    variable w
    variable mode
    variable zoom-mode

    variable prevcursor

    constructor { args } {
	set initialized 0
	set val(0) 1
	set last 0
	set mode normal
    }

    destructor {
	delete object  $w.diagram
    }

    method ui { n m t u g } {
	set menu $m
	set tb $t
	set ui $u
	set ogl $g

	set name $n
    }

    method init {} {

	#
	# widgets menu
	#

	$menu add menubutton .widgets -text "Widgets"

	$menu add command .widgets.hairline -label Hairline -underline 0 \
	    -command "$this widget hairline"
	$menu add command .widgets.axes -label Axes -underline 0 \
	    -command "$this widget axes"
	$menu add command .widgets.zoom -label Zoom -underline 0 \
	    -command "$this widget zoom"

	#
	# toolbar
	#

	$tb add button normal \
	    -helpstr "Normal mode" \
	    -image [image create photo \
		-file "../src/pixmaps/24x24-pointer.ppm"] \
	    -command "$this set-mode normal"

	$tb add button zoom-in \
	    -helpstr "Zoom mode" \
	    -image [image create photo \
		-file "../src/pixmaps/24x24mag-glass.ppm"] \
	    -command "$this set-mode zoom"
	
#	$tb add button sub \
\#	    -helpstr "SubWindow" \
\#	    -command {puts "sub window"}


	#
	# option area
	#
	iwidgets::labeledframe $ui.d -labeltext $name -labelpos nw

	set w [$ui.d childsite]

	# list of polys
	frame $w.poly
	pack $w.poly -side left -anchor n

	   # two different options
	   frame $w.poly.all
	   frame $w.poly.one

	   # select the 'all' option
	   pack $w.poly.all 
	

	# select polys
	frame $w.s

  	  label $w.s.select -text "Select:"
	  set $this-select 2

	  frame $w.s.b1
	  radiobutton $w.s.b1.one -text "One" \
	      -variable $this-select -value 1 \
	      -command "$this select one"
	  radiobutton $w.s.b1.many -text "Many" \
	      -variable $this-select -value 2 \
	      -command "$this select many"

	  pack $w.s.b1.one $w.s.b1.many -side left

	  label $w.s.scale -text "Scale:"
	  set $this-scale 1

	  frame $w.s.b2
	  radiobutton $w.s.b2.all -text "All" \
	      -variable $this-scale -value 1 \
	      -command "$this-c redraw" -anchor w
	  radiobutton $w.s.b2.each -text "Each" \
	      -variable $this-scale -value 2 \
	      -command "$this-c redraw" -anchor w
	  pack $w.s.b2.all $w.s.b2.each -side left
	
	pack $w.s.select -side top -anchor w
	pack $w.s.b1 -side top -anchor e
	pack $w.s.scale -side top -anchor w
	pack $w.s.b2 -side top -anchor e 

	pack $w.s -side left -ipadx 5 -anchor n

	pack $ui.d

	bind $name-DiagramTags <ButtonPress> "$this-c ButtonPress %x %y %b "
	bind $name-DiagramTags <B1-Motion> "$this-c Motion %x %y 1 "
	bind $name-DiagramTags <B2-Motion> "$this-c Motion %x %y 2 "
	bind $name-DiagramTags <B3-Motion> "$this-c Motion %x %y 3 "
	bind $name-DiagramTags <ButtonRelease> "$this-c ButtonRelease %x %y %b "

	bind DiagramZoom <KeyPress>   "$this zoom-key press %K; break" 
	bind DiagramZoom <KeyRelease> "$this zoom-key release %K; break"
	bind DiagramZoom <ButtonPress> "$this zoom-button %x %y %b; break "

	set initialized 1
    }
	
    method add { n name color} {
	if { $initialized == 0 } {
	    $this init
	}
	    
	checkbutton $w.poly.all.$name -text $name -variable val($name) \
	    -fg $color \
	    -command "$this-c select $n \$val($name)"
	$w.poly.all.$name select
	pack $w.poly.all.$name -side top

	radiobutton $w.poly.one.$name -text $name \
	    -variable select-one -value $n \
	    -bg $color \
	    -command "$this-c select-one $n"
	pack $w.poly.one.$name -side top -ipady 2

    }

    method select {which} {
	if { $which == "one" } {
	    pack forget $w.poly.all
	    pack $w.poly.one
	} else {
	    pack forget $w.poly.one 
	    pack $w.poly.all
	}
	$this-c redraw
    }

    method widget { name } {
	$this-c widget $name
    }

    method getbinds {} {
	puts "getbinds =  $name-DiagramTags"
	return $name-DiagramTags
    }

    method new-opt {} {
	set win $w.$last
	frame $win
	pack $win -side left -anchor n
	
	set last [expr $last + 1]
	return $win
    }

    method set-mode { new-mode } {
	if { ${new-mode} != $mode } {
	    if { ${new-mode} == "zoom" } {
		$this set-zoom on
	    } else {
		$this set-zoom off
	    }
	    set mode ${new-mode}
	}
    }

    method set-zoom { mode } {
	set icon "../src/pixmaps/viewmag"
	if { $mode == "on" } {
	    set prevcursor \
		[$ogl set-cursor "@$icon+.xbm $icon+mask.xbm black lightblue"]
	    $ogl add-bind DiagramZoom
	    set zoom-mode in
	} elseif { $mode == "off" } {
	    $ogl set-cursor $prevcursor
	    $ogl rem-bind DiagramZoom
	    set zoom-mode out
	} elseif { $mode == "in" } {
	    $ogl set-cursor "@$icon+.xbm $icon+mask.xbm black lightblue"
	    set zoom-mode in
	} else {
	    $ogl set-cursor "@$icon-.xbm $icon-mask.xbm black lightyellow"
	    set zoom-mode out
	}
    }

    method zoom-key { type key } {
	if { $key == "Control_L" | $key == "Control_R" } {
	    if { $type == "press" } {
		$this set-zoom out
	    } else {
		$this set-zoom in
	    }
	}
    }

    method zoom-button { x y b } {
	$this-c zoom ${zoom-mode} $x $y $b 
    }
}
