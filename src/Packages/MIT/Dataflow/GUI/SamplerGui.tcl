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
	set mode "stop"
    }

    method ui { window } {

	set w $window
	
        iwidgets::entryfield $w.iteration -labeltext "Iterations:" \
	    -validate numeric \
	    -textvariable $this-iterations \
	    -command "$this-c iteration \[$w.iteration get\]" 

#        iwidgets::entryfield $w.monitor -labeltext "Monitor:" \
\#	    -validate numeric -command "$this-c monitor \[$w.monitor get\]" 

#        iwidgets::entryfield $w.thin -labeltext "Thin:" \
\#	    -validate numeric -command "$this-c thin  \[$w.thin get\]" 

        iwidgets::entryfield $w.kappa -labeltext "Kappa:" \
	    -validate numeric -command "$this-c kappa \[$w.kappa get\]" 

	frame $w.ctrl
	button $w.ctrl.stop -text "Stop" -command "$this stop" -state disable
	button $w.ctrl.run -text "  Run   " -command "$this run "
	label  $w.ctrl.current -text ""
	pack $w.ctrl.stop $w.ctrl.run $w.ctrl.current -side left -anchor w

	frame $w.children

	pack $w.iteration -anchor w
	pack $w.ctrl -anchor w
	pack $w.children 

	set n 0
	#iwidgets::Labeledframe $w.graph -labeltext "Progress"
	#pack $w.graph -expand yes -fill both
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

    method new-child-window { name } {
#	set child [frame $w.children.$n]
#	pack $child -side top
#	incr n
#	return $child
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}




