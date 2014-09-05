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
#    File   : ShowLeads.tcl
#    Author : Martin Cole
#    Date   : Tue Jan  8 07:19:16 2002

catch {rename ShowLeads ""}

itcl_class BioPSE_Visualization_ShowLeads {
    inherit Module
    constructor {config} {
	set name ShowLeads
	set_defaults
    }
    method set_defaults {} {
	#current color
	global $this-r
	global $this-g
	global $this-b

	#position of range lines
	global $this-range1
	global $this-range2
	global $this-range-height

	#stored vals
	global $this-xvals
	global $this-yvals
	
	# vertical scale between leads
	global $this-vscale
	global $this-min
	global $this-max

	#units info for time
	global $this-time-units
	global $this-time-min
	global $this-time-max

	set $this-r 20
	set $this-g 45
	set $this-b 100
	set $this-range1 0
	set $this-range2 0
	set $this-range-height 10
	set $this-xvals {}
	set $this-yvals {}
	set $this-vscale 0.125
	set $this-min 0.0
	set $this-max 10.0
	set $this-time-units "s"
	set $this-time-min .1
	set $this-time-max .789

    }
    
    method scale_graph {val} {
	global $this-vscale
	global $this-min
	global $this-max
	set w .ui[modname]
	
	set min [set $this-min]
	set max [set $this-max]
	if {$min >= $max} {
	    set min [expr $max - 1.0]
	}

	set min [expr [set $this-vscale] * $min]
	set max [expr [set $this-vscale] * $max]
	$w.graph axis configure y -min $min -max $max
    }
    
    method min_changed {} {
	global $this-min
	set w .ui[modname]
	
	set min [expr [set $this-min] + 1]
	$w.np.max configure -from $min
    }

    method max_changed {} {
	global $this-max
	set w .ui[modname]
	
	set max [expr [set $this-max] - 1]
	$w.np.min configure -to $max
    }
    
    method ui {} {
	global $this-vscale
	global $this-min
	global $this-max

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
	toplevel $w
	wm minsize $w 450 200
	set n "$this-c needexecute "
	
	button $w.execute -text "Execute" -command $n
	pack $w.execute -side top -fill x -pady 2 -padx 2
        
	frame $w.np
	pack $w.np -side top
	
	scale $w.np.min -label "Show lead from:" \
		-orient horizontal -variable $this-min \
		-command "$this scale_graph"
	scale $w.np.max -label "to:" \
		-orient horizontal -variable $this-max \
		-command "$this scale_graph"
	
	
	bind $w.np.min <ButtonRelease-1> "$this min_changed"
	bind $w.np.max <ButtonRelease-1> "$this max_changed"

	pack $w.np.min $w.np.max -side left

	expscale $w.slide -label "Vertical Scale" \
		-orient horizontal -variable $this-vscale	
	bind $w.slide.scale <ButtonRelease-1> "$this redraw_leads"

	blt::graph $w.graph -title "Leads" -height 450 \
		-plotbackground black
	$w.graph yaxis configure -title ""  
	$w.graph xaxis configure -title "" -loose true
	bind $w.graph <ButtonPress-1> "$this select_range %x %y"
	bind $w.graph <Button1-Motion> "$this move_range %x %y"
	bind $w.graph <ButtonRelease-1> "$this deselect_range %x %y"

	#puts [$w.graph axis names]
	$w.graph axis configure y  -command "$this draw_yaxis"
	$w.graph axis configure x  -command "$this draw_xaxis"

	set r1 [set $this-range1]
	set r2 [set $this-range2]
	if {! [$w.graph element exists "range1"]} {
	    $w.graph element create "range1" -linewidth 1 -color yellow\
		    -data {$r1 -1 $r1 10} -pixels 3 -label ""
	
	    $w.graph element create "range2" -linewidth 1 -color yellow\
		    -data {$r2 -1 $r2 10} -pixels 3 -label ""
	}

	draw_leads
	pack $w.graph
    }
    
    # time axis
    method draw_xaxis {arg arg1} {
	set units [set $this-time-units]
	set t0 [set $this-time-min]
	set t1 [set $this-time-max]
	set t [expr ($arg1 * ($t1-$t0)) + $t0]
	return [format "%.2f%s" $t $units]
    }
    method draw_yaxis {arg arg1} {
	set l [expr $arg1 / [set $this-vscale]]
	return [format "%.1f" $l]
    }
    method get_color {} {
	global $this-r
	global $this-g
	global $this-b
	
	set r [set $this-r]
	set g [set $this-g]
	set b [set $this-b]

	set r [expr $r + 85]
	set g [expr $g + 85]
	set b [expr $b + 85]

	if {[expr $r >= 255]} {set r [expr $r % 255 + 11]}
	if {[expr $g >= 255]} {set g [expr $g % 255 + 13]}
	if {[expr $b >= 255]} {set b [expr $b % 255 + 17]}

	set $this-r $r
	set $this-g $g
	set $this-b $b

	return [format "#%02x%02x%02x" $r $g $b]
    }

    method redraw_leads {} {
	global $this-xvals
	global $this-yvals
	global $this-vscale
	set w .ui[modname]
	
	set ydat [set $this-yvals]
	set leads [llength $ydat]
	set xvals [lindex [set $this-xvals] 0]

	for {set ind 0} {$ind < $leads} {incr ind} {
	    set yvals [lindex $ydat $ind]
	    # offest the values so that they fit in their own space in the graph
	    set offset [expr [set $this-vscale] * $ind]
	    set ovals {}
	    set zlvals {}
	    #puts [lindex $yvals 0]
	    #return
	    for {set i 0} { $i < [llength $yvals] } { incr i } {
		lappend ovals [expr [lindex $yvals $i] + $offset] 
		lappend zlvals $offset
	    }

	    # add the element to the graph
	    set name [format "%s%d" "lead-" $ind]
	    blt::vector xvec yvec zlvec
	    set xvec $xvals
	    set yvec $ovals
	    set zlvec $zlvals
	    $w.graph element configure $name -linewidth 1 -label ""\
		    -xdata $xvec -ydata $yvec -symbol ""
	    # add 0 line
	    set zl [format "%s%s" $name "-zl"]
	    $w.graph element configure $zl -linewidth .5 \
		    -xdata $xvec -ydata $zlvals -symbol "" -label "" \
		    -color #414141 -dashes { 10 2 1 2 }
	    
	    #add markers [lindex $yvals 0] 
	    set mk [format "%s%s" $name "-mk"]
	    set v 0.05
	    set t [format "     %+.2f" [lindex $yvals 0]]
	    $w.graph marker configure $mk -coords {$v $offset} \
		    -text $t
	}
	#add the range bars
	set $this-range-height [expr [set $this-vscale] * [expr $leads + 2]]
	set top [set $this-range-height]
	set bot [expr [set $this-vscale] * -1.0]
	set r1 [set $this-range1]
	set r2 [set $this-range2]
	$w.graph element configure "range1" -data {$r1 $bot $r1 $top} 
	$w.graph element configure "range2" -data {$r2 $bot $r2 $top} 

	#configure the min and max sliders
	$w.np.min configure  -to [expr [set $this-max] - 1]
	$w.np.max configure -from  [expr [set $this-min] + 1] -to $leads
	#force a redraw
	$this scale_graph 1.
    }

    method draw_leads {} {
	global $this-xvals
	global $this-yvals
	global $this-vscale
	set w .ui[modname]
	set ydat [set $this-yvals]
	set leads [llength $ydat]
	set xvals [lindex [set $this-xvals] 0]

	for {set ind 0} {$ind < $leads} {incr ind} {
	    set yvals [lindex $ydat $ind]
	    # offest the values so that they fit in their own space in the graph
	    set offset [expr [set $this-vscale] * $ind]
	    set ovals {}
	    set zlvals {}
	    #puts [lindex $yvals 0]
	    #return
	    for {set i 0} { $i < [llength $yvals] } { incr i } {
		lappend ovals [expr [lindex $yvals $i] + $offset] 
		lappend zlvals $offset
	    }
	    set col [get_color]
	    # add the element to the graph
	    set name [format "%s%d" "lead-" $ind]
	    blt::vector xvec yvec zlvec
	    set xvec $xvals
	    set yvec $ovals
	    set zlvec $zlvals
	    if {[$w.graph element exists $name]} {
		$w.graph element delete $name
	    }
	    $w.graph element create $name -linewidth 1 -label ""\
		    -xdata $xvec -ydata $yvec -symbol "" -color $col
	    # add 0 line
	    set zl [format "%s%s" $name "-zl"]
	    if {[$w.graph element exists $zl]} {
		$w.graph element delete $zl
	    }
	    $w.graph element create $zl -linewidth .5 \
		    -xdata $xvec -ydata $zlvals -symbol "" -label "" \
		    -color #414141 -dashes { 10 2 1 2 }
	    
	    #add markers [lindex $yvals 0] 
	    set mk [format "%s%s" $name "-mk"]
	    set v 0.05
	    set t [format "     %+.2f" [lindex $yvals 0]]
	    $w.graph marker create text -name $mk \
		    -coords {$v $offset} \
		    -text $t -background "" -foreground $col
	}
	#add the range bars
	set $this-range-height [expr [set $this-vscale] * [expr $leads + 2]]
	set top [set $this-range-height]
	set bot [expr [set $this-vscale] * -1.0]
	set r1 [set $this-range1]
	set r2 [set $this-range2]

	$w.graph element configure "range1" -data {$r1 $bot $r1 $top} 
	$w.graph element configure "range2" -data {$r2 $bot $r2 $top} 

	#configure the min and max sliders
	$w.np.min configure  -to [expr [set $this-max] - 1]
	$w.np.max configure -from  [expr [set $this-min] + 1] -to $leads
    }

    method add_lead {ind xvals yvals} {
	global $this-xvals
	global $this-yvals
	set w .ui[modname]
	
	# cache off vals
	if {[llength [set $this-xvals]] == 0} {lappend $this-xvals $xvals}
	lappend $this-yvals $yvals

    }

    # Given the index a range marker is over, update the value markers for 
    # each lead.
    method update_markers {ind} {

	global $this-xvals
	global $this-yvals
	set w .ui[modname]
	set yvals [set $this-yvals]
	for {set i 0} { $i < [llength $yvals] } { incr i } {
	    set curyvals [lindex $yvals $i]
	    set marker [format "lead-%d-mk" $i]
	    set t [format "     %+.2f" [lindex $curyvals $ind]]
	    $w.graph marker configure $marker -text $t
	}
    }
    
    protected range_selected false

    method select_range {wx wy} {
	global $this-range1
	global $this-range2
	set w .ui[modname]
	
	#transform current index for range into window coordinates
	set pos [$w.graph transform [set $this-range1] 0]
	set r1posx [lindex $pos 0]
	set pos [$w.graph transform [set $this-range2] 0]
	set r2posx [lindex $pos 0]

	if {abs($wx - $r2posx) < 5} {
	    $w.graph element configure "range2" -color red
	    set range_selected range2
	} else {
	    if {abs($wx - $r1posx) < 5} {
		$w.graph element configure "range1" -color red
		set range_selected range1
	    }
	}
    }

    # Given a data value in the xvalues range, find the index closest to
    # that value.
    method get_xind {val} {
	global $this-xvals
	set xvals [lindex [set $this-xvals] 0]
	set closest 0
	set closestval [expr abs($val - [lindex $xvals 0])]
	for {set i 1} {$i < [llength $xvals]} {incr i} {
	    set cur [expr abs($val - [lindex $xvals $i])]
	    if {$cur < $closestval} {
		set closest $i
		set closestval [expr abs($val - [lindex $xvals $i])]
	    }
	}
	set l {}
	lappend l $closest
	lappend l [lindex $xvals $closest]
	return $l
    }
    
    method move_range {wx wy} {
	if {$range_selected != "false"} {

	    set top [set $this-range-height]
	    set bot [expr [set $this-vscale] * -1.0]
	    set w .ui[modname]
	    set newindex [lindex [$w.graph invtransform $wx $wy] 0]
	    set n [get_xind $newindex]
	    set x [lindex $n 1]
	    $w.graph element configure $range_selected -data {$x $bot $x $top}
	    update_markers [lindex $n 0]
	} 
    }
    
    method deselect_range {wx wy} {
	if {$range_selected != "false"} {
	    set w .ui[modname]
	    set top [set $this-range-height]
	    set bot [expr [set $this-vscale] * -1.0]
	    $w.graph element configure $range_selected -color yellow
	    set newindex [lindex [$w.graph invtransform $wx $wy] 0]
	    set n [get_xind $newindex]
	    set x [lindex $n 1]
	    $w.graph element configure $range_selected -data {$x $bot $x $top}
	    update_markers [lindex $n 0]

	    if {$range_selected == "range1"} {
		global $this-range1
		set $this-range1 $newindex
	    } else {
		global $this-range2
		set $this-range2 $newindex
	    }
	    set range_selected false
	}
    }

}
