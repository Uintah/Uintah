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


package require Iwidgets 3.1

class SamplerGui {

    variable w
    variable n
    variable mode 

    constructor {} {
	global $this-interations
	global $this-sub
	global $this-kappa

	set mode "stop"
	set n 0
    }

    method ui { window } {
	global $this-interations
	global $this-sub
	global $this-kappa

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

	# Kappa
        iwidgets::entryfield $w.kappa -labeltext "Kappa:" \
	    -validate numeric -width 6 \
	    -textvariable $this-kappa \
	    -command "$this-c kappa \[$w.kappa get\]" 
	pack $w.kappa -anchor w

	# Control
	frame $w.ctrl
	button $w.ctrl.stop -text "Stop" -command "$this stop" -state disable
	button $w.ctrl.run -text "  Run   " -command "$this run "
	label  $w.ctrl.current -text ""
	pack $w.ctrl.stop $w.ctrl.run $w.ctrl.current -side left -anchor w
	pack $w.ctrl -anchor w

	# Children
	frame $w.children
	pack $w.children 
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
	
    method new-child-window { name } {
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}




