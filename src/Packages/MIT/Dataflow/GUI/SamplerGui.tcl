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


package require Iwidgets 3.0

itcl::class SamplerGui {

    variable w
    variable n
    variable mode

    variable displayed

    constructor {} {
	global $this-iterations
	global $this-sub
	global $this-kappa
	global $this-nparms
	global $this-maxnparms

	set $this-nparms 0
	set mode "stop"
	set n 0
	set $this-maxnparms 0

	set displayed 0
    }

    method ui { window } {
	global $this-iterations
	global $this-sub
	global $this-kappa
	global $this-nparms

	set w $window
	
	# Iterations & Subsampling
	frame $w.set
        iwidgets::entryfield $w.set.iteration -labeltext "Iterations:" \
	    -validate numeric -width 6 \
	    -textvariable $this-iterations \
	    -command "$this-c iterations \[$w.set.iteration get\]" 

        iwidgets::entryfield $w.set.sub -labeltext "Subsample:" \
	    -validate numeric -width 3 \
	    -textvariable $this-sub \
	    -command "$this-c sub \[$w.set.sub get\]" 

	pack $w.set.iteration $w.set.sub -anchor w -side left
	pack $w.set -anchor w
	
	frame $w.greek
	# Kappa, Sigma & Theta
        iwidgets::entryfield $w.greek.kappa -labeltext "Kappa:" \
	    -validate numeric -width 6 \
	    -textvariable $this-kappa \
	    -command "$this-c kappa \[$w.kappa get\]" 

	iwidgets::scrolledframe $w.greek.theta -labeltext "Theta" \
		-borderwidth 2 \
		-relief flat -width 475 -height 65

	iwidgets::scrolledframe $w.greek.sigma -labeltext "Sigma" \
		-borderwidth 2 \
		-relief flat -width 475 -height 150

	pack $w.greek.kappa -anchor w

	button $w.greek.showsigma -text "Show Sigma" -command "$this sigma 1"
	button $w.greek.showtheta -text "Show Theta" -command "$this theta 1"
	pack $w.greek.showtheta $w.greek.showsigma -anchor w
	pack $w.greek -anchor w

	# Control
	frame $w.ctrl
	button $w.ctrl.stop -text "Stop" -command "$this stop" -state disable
	button $w.ctrl.run -text "  Run   " -command "$this run "
	label  $w.ctrl.current -text ""
	pack $w.ctrl.stop $w.ctrl.run $w.ctrl.current -side left -anchor w
	pack $w.ctrl -anchor w
	
	# Children
	frame $w.children
	pack $w.children  -expand true -fill both -anchor nw

    }

    method run {} {
	if { $mode == "stop" } {
	    $w.ctrl.run configure -text " Pause "
	    $w.ctrl.stop configure -state normal
	    set mode "run"
	    $this-c run
	} elseif { $mode == "run" } {
	    $w.ctrl.run configure -text "Continue"
	    set mode "pause"
	    $this-c pause
	} else {
	    $w.ctrl.run configure -text " Pause  "
	    set mode "run"
	    $this-c run
	}
    }

	
    method stop {} {
	$w.ctrl.run configure -text "Run"
	$w.ctrl.stop configure -state disable
	set mode "stop"
	$this-c stop
    }
    
    
    method done {} {
	if { $mode == "run" } {
	    $w.ctrl.run configure -text "Run"
	    $w.ctrl.stop configure -state disable
	    set mode "stop"
	}
    }
    
    method set-iter { n } {
	$w.ctrl.current configure -text $n
    }

    method set-kappa { k } {
	global $this-kappa
	set $this-kappa $k
    }

    method set-nparms { val } {
	global $this-nparms
	set $this-nparms $val
	$this sigmathetawidgets
    }

    method set-sigma { args } {

    }

    method set-theta { args } {
	for {set i 0} {$i < [set $this-nparms]} {incr i} {
	    set $this-theta($i) [lindex $args $i]
	}

    }
    method send-theta {} {
	set THETA "$this-c theta [set $this-nparms]"
	for {set i 0} {$i < [set $this-nparms]} {incr i} {
	    append THETA " [set $this-theta($i)]"
	}
	eval $THETA
    }

    method send-sigma {} {
	puts "SIGMA"
    }

    method sigmathetawidgets {} {
	set sigma [$w.greek.sigma childsite]
	set theta [$w.greek.theta childsite]
	
	if {$displayed == 0} {
	    label $sigma.overall -text "Row#\\Col#" -relief sunken -width 10 \
		    -justify right
	    label $theta.overall -text "Column #" -relief sunken\
		    -width 10 -justify right
	    label $theta.label -text "Value" -relief sunken -width 10 \
		    -justify right

	    button $sigma.update -text "Update" -command "$this send-sigma"
	    button $theta.update -text "Update" -command "$this send-theta"

	    grid config $sigma.update -column 0 -row 0 
	    grid config $theta.update -column 0 -row 0
	    grid config $sigma.overall -column 1 -row 0 -sticky sn
	    grid config $theta.overall -column 1 -row 0 -sticky sn
	    grid config $theta.label -column 1 -row 1 -sticky sn

	    set displayed 1
	}

	for {set i [set $this-maxnparms]} {$i < [set $this-nparms]} {incr i} {
	    label $sigma.row[set i] -text "$i" -relief sunken -width 10 \
		    -justify right
	    label $sigma.col[set i] -text "$i" -relief sunken

	    label $theta.col[set i] -text "$i" -relief sunken

	    iwidgets::entryfield $theta.c[set i] \
		    -validate numeric -width 8 \
		    -textvariable $this-theta([set i])

	    grid config $sigma.col[set i] \
		    -column [expr $i + 2] -row 0 -sticky snew
	    grid config $sigma.row[set i] \
		    -column 1 -row [expr $i + 1] -sticky sn
	    grid config $theta.col[set i] \
		    -column [expr $i + 2] -row 0 -sticky snew
	
	    for {set j [set $this-maxnparms]} {$j < [set $this-nparms]} {incr j} {
		iwidgets::entryfield $sigma.r[set i]c[set j] \
			-validate numeric -width 8 \
			-textvariable $this-blah
	    }
	}

	for {set i 0} {$i < [set $this-nparms]} {incr i} {
	    grid config $theta.c[set i] \
		    -column [expr $i + 2] -row 1
	    for {set j 0} {$j < [set $this-nparms]} {incr j} {
		grid config $sigma.r[set i]c[set j] \
			-column [expr $j + 2] -row [expr $i + 1]
	    }
	}
	if { [set $this-nparms] > [set $this-maxnparms] } {
	    set $this-maxnparms [set $this-nparms]
	}
    }	

    method sigma {val} {
	if {$val} {
	    $this sigmathetawidgets
	    pack $w.greek.sigma
	    $w.greek.showsigma configure -text "Hide Sigma"
	    $w.greek.showsigma configure -command "$this sigma 0" 
	} else {	
	    set sigma [$w.greek.sigma childsite]
	    $w.greek.showsigma configure -text "Show Sigma"
	    $w.greek.showsigma configure -command "$this sigma 1"
	    pack forget $w.greek.sigma	
	    for {set i 0} {$i < [set $this-nparms]} {incr i} {
		for {set j 0} {$j < [set $this-nparms]} {incr j} {
		    grid forget $sigma.r[set i]c[set j]
		}
	    }

	}
    }

    method theta {val} {
	if {$val} {
	    $this sigmathetawidgets
	    pack $w.greek.theta
	    $w.greek.showtheta configure -text "Hide Theta"
	    $w.greek.showtheta configure -command "$this theta 0"
	} else {
	    $w.greek.showtheta configure -text "Show Theta"
	    $w.greek.showtheta configure -command "$this theta 1"
	    pack forget $w.greek.theta	

	    set theta [$w.greek.theta childsite]
	    for {set i 0} {$i < [set $this-nparms]} {incr i} {
		grid forget $theta.c[set i] 
	    }
	}
	
    }

    method new-child-window { name } {
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w -expand true -fill both
	incr n
	return [$child childsite]
    }
}




