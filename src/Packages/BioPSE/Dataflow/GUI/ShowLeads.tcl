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
    
    protected range_selected false

    method set_defaults {} {
	#current color
	global $this-r
	global $this-g
	global $this-b

	#position of range lines
	global $this-range1
	global $this-range2
	global $this-range-top
	global $this-range-bot

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
	set $this-range-top 10
	set $this-range-bot -1
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
	    set $this-min $max
	}

	clear_graph
	draw_leads
    }
    
    method min_changed {} {
	global $this-min
	set w .ui[modname]
	
	set min [set $this-min]
	$w.np.max configure -from $min
    }

    method max_changed {} {
	global $this-max
	set w .ui[modname]
	
	set max [set $this-max]
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

	$w.np.min configure  -from 0 -to [set $this-max]
	$w.np.max configure -from  [set $this-min] -to [set $this-max]
	
	bind $w.np.min <ButtonRelease-1> "$this min_changed"
	bind $w.np.max <ButtonRelease-1> "$this max_changed"

	pack $w.np.min $w.np.max -side left

	expscale $w.slide -label "Vertical Scale" \
		-orient horizontal -variable $this-vscale	
	bind $w.slide.scale <ButtonRelease-1> "$this draw_leads"

	blt::graph $w.graph -title "Leads" -height 450 \
		-plotbackground black
	$w.graph yaxis configure -title ""  
	$w.graph xaxis configure -title "" -loose true
	bind $w.graph <ButtonPress-1> "$this select_range %x %y"
	bind $w.graph <Button1-Motion> "$this move_range %x %y"
	bind $w.graph <ButtonRelease-1> "$this deselect_range %x %y"

	##puts [$w.graph axis names]
	$w.graph axis configure y  -command "$this draw_yaxis"
	$w.graph axis configure x  -command "$this draw_xaxis"

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
	return "$arg1"
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

    method set_min_max_index {min max} {
	set w .ui[modname]

	if {[winfo exists $w.np.min]} {
	    #configure the min and max sliders
	    $w.np.min configure  -from 0 -to [expr $max]
	    $w.np.max configure -from  $min -to [expr $max]
	}

	set $this-min $min
	set $this-max $max
    }

    method draw_leads {} {
	set w .ui[modname]
	if {! [winfo exists $w]} {
	    return;
	}
	
	global $this-xvals
	global $this-yvals
	global $this-vscale
	set w .ui[modname]
	set ydat [set $this-yvals]
	set leads [llength $ydat]
	if {$leads <= 0} { return }


	set xvals [lindex [set $this-xvals] 0]
	set begin [expr int([set $this-min])]
	set end [expr int([set $this-max])]
	set absmin [lindex [lindex $ydat $begin] 0]
	set absmax [lindex [lindex $ydat $begin] 0]
	set offset_idx 0
	for {set ind $begin} {$ind <= $end} {incr ind} {
	    #puts $ind
	    set yvals [lindex $ydat $ind]
	    #puts $yvals
	    # offest the values so that they fit in their own 
	    # space in the graph
	    
	    set offset [expr [set $this-vscale] * $offset_idx]

	    set ovals {}
	    set zlvals {}
	    ##puts [lindex $yvals 0]
	    #return
	    for {set i 0} { $i < [llength $yvals] } { incr i } {
		set val [expr [lindex $yvals $i] + $offset]
		lappend ovals $val
		lappend zlvals $offset
		#puts $val
		if {$val > $absmax} { set absmax $val } 
		if {$val < $absmin} { set absmin $val } 
	    }
	    set col [get_color]
	    # add the element to the graph
	    set name [format "%s%d" "lead-" $ind]
	    blt::vector create xvec$ind 
	    blt::vector create yvec$ind
	    blt::vector create zlvec$ind

	    xvec$ind set $xvals
	    yvec$ind set $ovals
	    zlvec$ind set $zlvals
	    if {[$w.graph element exists $name]} {
		$w.graph element configure $name -linewidth 1 -label ""\
		    -xdata xvec$ind -ydata yvec$ind -symbol ""
	    } else {
		$w.graph element create $name -linewidth 1 -label ""\
		    -xdata xvec$ind -ydata yvec$ind -symbol "" -color $col
	    }
	    # add 0 line
	    set zl [format "%s%s" $name "-zl"]
	    if {[$w.graph element exists $zl]} {
		$w.graph element configure $zl -linewidth .5 \
		    -xdata xvec$ind -ydata zlvec$ind -symbol "" -label ""
	    } else {
		$w.graph element create $zl -linewidth .5 \
		    -xdata xvec$ind -ydata zlvec$ind -symbol "" -label "" \
		    -color #414141 -dashes { 10 2 1 2 }
	    }
	    #add markers [lindex $yvals 0] 
	    set mk [format "%s%s" $name "-mk"]
	    set v 0.05
	    set val [lindex $yvals 0]
	    if {$val == ""} { set val "0.0" }
	    set t [format "     %+.2f" $val]
	    if {[$w.graph marker exists $mk]} {
		$w.graph marker configure $mk -coords {$v $offset} -text $t
	    } else {
		$w.graph marker create text -name $mk \
		    -coords {$v $offset} \
		    -text $t -background "" -foreground $col
	    }
	    incr offset_idx
	}

	#set yscale to the appropriate extents
	if {$absmax == ""} {
	    set min -1
	    set max 10
	} else {
	    set pad [expr ( $absmax - $absmin ) * 0.1] 
	    set min [expr $absmin - $pad]
	    set max [expr $absmax + $pad]
	}
	#puts "min $min max $max"
	if {$max <= $min} {set max [expr $min + 1]}
	$w.graph axis configure y -min $min -max $max

	#add the range bars
	set $this-range-top $max
	set $this-range-bot $min
	set top [set $this-range-top]
	set bot [set $this-range-bot]
	set r1 [set $this-range1]
	set r2 [set $this-range2]

	if {[$w.graph element exists "range1"]} {
	    $w.graph element configure "range1" -data {$r1 $bot $r1 $top} 
	    $w.graph element configure "range2" -data {$r1 $bot $r1 $top} 
	} else {
	    $w.graph element create "range1" -data {$r1 $bot $r1 $top} \
		-color yellow -pixels 3 -label ""
	    $w.graph element create "range2" -data {$r2 $bot $r2 $top} \
		-color yellow -pixels 3 -label ""
	}
    }

    method add_lead {ind xvals yvals} {
	global $this-xvals
	global $this-yvals
	set w .ui[modname]

	# cache off vals
	if {[llength [set $this-xvals]] == 0} {lappend $this-xvals $xvals}
	lappend $this-yvals $yvals

	#puts [set $this-xvals]
	#puts [set $this-yvals]

    }

    # Given the index a range marker is over, update the value markers for 
    # each lead.
    method update_markers {ind} {

	global $this-xvals
	global $this-yvals
	set w .ui[modname]
	set yvals [set $this-yvals]

	set begin [expr int([set $this-min])]
	set end [expr int([set $this-max])]
	for {set i $begin} {$i <= $end} {incr i} {
	    set curyvals [lindex $yvals $i]
	    set marker [format "lead-%d-mk" $i]
	    set t [format "     %+.2f" [lindex $curyvals $ind]]
	    $w.graph marker configure $marker -text $t
	}
    }

    method clear_graph {} {

	set w .ui[modname]
	set yvals [set $this-yvals]
	set leads [llength $yvals]
	
	if {[winfo exists $w.graph]} {
	    for {set i 0} {$i < $leads} {incr i} {
		if {[winfo exists xvec$i]} {
		    blt::vector destroy xvec$i
		}
		if {[winfo exists yvec$i]} {
		    blt::vector destroy yvec$i
		}
		if {[winfo exists zlvec$i]} {
		    blt::vector destroy zlvec$i
		}
		set name [format "%s%d" "lead-" $i]
		
		if {[$w.graph element exists $name]} {
		    $w.graph element delete $name
		}
		set zl [format "%s%s" $name "-zl"]
		
		if {[$w.graph element exists $zl]} {
		    $w.graph element delete $zl
		}
		
		set mk [format "%s%s" $name "-mk"]
		
		if {[$w.graph marker exists $mk]} {
		    $w.graph marker delete $mk
		}
	    }
	}
    }

    method clear {} {
	set_defaults
	clear_graph
    }
       
    
    method select_range {wx wy} {
	global $this-range1
	global $this-range2
	set w .ui[modname]
	#puts select_range 
	#puts $range_selected
	#transform current index for range into window coordinates
	set pos [$w.graph transform [set $this-range1] 0]
	set r1posx [lindex $pos 0]
	set pos [$w.graph transform [set $this-range2] 0]
	set r2posx [lindex $pos 0]
	
	#puts "r1x is $r1posx"
	#puts "r2x is $r2posx"
	#puts "wx is $wx"

	if {abs($wx - $r2posx) < 5} {
	    $w.graph element configure "range2" -color red
	    set range_selected range2
	} else {
	    if {abs($wx - $r1posx) < 5} {
		$w.graph element configure "range1" -color red
		set range_selected range1
	    } else {
		set range_selected false
	    }
	}
    }

    # Given a data value in the xvalues range, find the index closest to
    # that value.
    method get_xind {val} {
	global $this-xvals
	if { [set $this-xvals] == "" } { return 0 }
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
	#puts move_range 
	#puts $range_selected
	if {$range_selected != "false"} {
	    #puts "move_range doing work"
	    set top [set $this-range-top]
	    set bot [set $this-range-bot]
	    set w .ui[modname]
	    set newindex [lindex [$w.graph invtransform $wx $wy] 0]
	    set n [get_xind $newindex]
	    set x [lindex $n 1]
	    $w.graph element configure $range_selected -data {$x $bot $x $top}
	    update_markers [lindex $n 0]
	} 
    }
    
    method deselect_range {wx wy} {
	#puts deselect_range
	#puts $range_selected
	if {$range_selected != "false"} {
	    #puts "deselect_range doin work"
	    set w .ui[modname]
	    set top [set $this-range-top]
	    set bot [set $this-range-bot]
	    $w.graph element configure $range_selected -color yellow
	    set newindex [lindex [$w.graph invtransform $wx $wy] 0]
	    set n [get_xind $newindex]
	    set x [lindex $n 1]
	    #puts "setting new range index to $x"
	    $w.graph element configure $range_selected -data {$x $bot $x $top}
	    update_markers [lindex $n 0]

	    if {$range_selected == "range1"} {
		global $this-range1
		set $this-range1 $x
	    } else {
		global $this-range2
		set $this-range2 $x
	    }
	}
	set range_selected false
    }

}
