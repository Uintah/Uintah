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

itcl_class SCIRun_FieldsOther_ScalarFieldStats {
    inherit Module
    protected draw_graph_needed 0
    constructor {config} {
        set name ScalarFieldStats

	global $this-min
	global $this-max
	global $this-mean
	global $this-median
	global $this-sigma
	global $this-is_fixed
	global $this-nbuckets

        set_defaults
    }

    method set_defaults {} {
	set $this-min "?"
	set $this-max "?"
	set $this-mean "?"
	set $this-median "?"
	set $this-sigma "?"
	set $this-is_fixed 0
	set $this-nbuckets 256
   }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	set n "$this-c needexecute "

        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Range"  -variable $this-is_fixed \
		-value 0 -command "$this autoRange"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Range"  -variable $this-is_fixed \
		-value 1 -command "$this fixedRange"
        pack $w.f2.b -side left

	frame $w.row1
	frame $w.row2
	frame $w.row3
	frame $w.row4

	pack $w.row1 $w.row2 $w.row3 $w.row4 \
	    -side top -e y -f both -padx 5 
	
	label $w.row1.min_label -text "Min Value:  "
	entry $w.row1.min_value -textvariable $this-min
	label $w.row1.max_label -text ",  Max Value:  "
	entry $w.row1.max_value -textvariable $this-max
	pack $w.row1.min_label $w.row1.min_value \
	    $w.row1.max_label $w.row1.max_value -side left

        bind $w.row1.min_value <Return> $n
        bind $w.row1.max_value <Return> $n

	label $w.row2.mean_label -text "Mean:  "
	label $w.row2.mean_value -textvariable $this-mean
	pack $w.row2.mean_label $w.row2.mean_value -side left

	label $w.row3.median_label -text "Median:  "
	label $w.row3.median_value -textvariable $this-median
	pack $w.row3.median_label $w.row3.median_value -side left

	label $w.row4.sigma_label -text "Standard Deviation:  "
	label $w.row4.sigma_value -textvariable $this-sigma
	pack $w.row4.sigma_label $w.row4.sigma_value -side left

	blt::barchart $w.graph -title "Histogram" \
	    -height [expr [set $this-nbuckets]*3/4.0] \
	    -width [set $this-nbuckets] -plotbackground gray80
	pack $w.graph

	frame $w.size -relief flat
	pack $w.size -side top -expand yes -fill x
	label $w.size.l -text "Histogram Bins:  "
	entry $w.size.e -textvariable $this-nbuckets
	pack $w.size.l $w.size.e -side left -expand yes 

	bind $w.size.e <Return> "$this resize_graph; $n"

	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -expand yes -fill x


       if { [set $this-is_fixed] } {
            $w.f2.b select
            $this fixedRange
        } else {
            $w.f1.b select
	    global $this-is_fixed
	    set w .ui[modname]
	    
	    set $this-is_fixed 0
	    
	    set color "#505050"
	    
	    $w.row1.min_label configure -foreground $color
	    $w.row1.min_value configure -state disabled -foreground $color
	    $w.row1.max_label configure -foreground $color
	    $w.row1.max_value configure -state disabled -foreground $color
        }

	if { $draw_graph_needed } {
	    $this-c needexecute
	}

    }

    method resize_graph { } {
	global $this-nbuckets
	set w .ui[modname]
	
	$w.graph configure -width [set $this-nbuckets]
	$w.graph configure -height [expr [set $this-nbuckets]*3/4.0]
    }

    method autoRange { } {
        global $this-is_fixed
        set w .ui[modname]
        
        set $this-is_fixed 0

        set color "#505050"

        $w.row1.min_label configure -foreground $color
        $w.row1.min_value configure -state disabled -foreground $color
        $w.row1.max_label configure -foreground $color
        $w.row1.max_value configure -state disabled -foreground $color
	$this-c needexecute	

    }

    method fixedRange { } {
        global $this-is_fixed
        set w .ui[modname]

        set $this-is_fixed 1


        $w.row1.min_label configure -foreground black
        $w.row1.min_value configure -state normal -foreground black
        $w.row1.max_label configure -foreground black
        $w.row1.max_value configure -state normal -foreground black
        
    }

    method graph_data { nmin nmax args } {
	global $this-min
	global $this-min
	
	set w .ui[modname]
	if {[winfo exists $w.graph] != 1} {
	    set draw_graph_needed 1
	    return
	} else {
	    set draw_graph_needed 0
	}
	
	if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
	    $w.graph axis configure y -logscale yes
	} else {
	    $w.graph axis configure y -logscale no
	}

	set min [set $this-min]
	set max [set $this-max]
	set xvector {}
	set yvector {}
	set yvector [concat $yvector $args]
	set frac [expr double(1.0/[llength $yvector])]

	$w.graph configure -barwidth $frac
	$w.graph axis configure x -min $min -max $max -subdivisions 4 -loose 1

	for {set i 0} { $i < [llength $yvector] } {incr i} {
	    set val [expr $min + $i*$frac*($max-$min)]
	    lappend xvector $val
	}
	
#	lappend yvector [split $args]

	if { [$w.graph element exists "h"] == 1 } {
	    $w.graph element delete "h"
	}

	$w.graph element create "h" -xdata $xvector -ydata $yvector
	    
    }
}


