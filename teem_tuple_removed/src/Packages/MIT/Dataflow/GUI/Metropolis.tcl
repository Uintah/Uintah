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


catch {rename Metropolis ""}

#package require SCIRun_Graph 1.0
package require Iwidgets 3.0

itcl_class MIT_Bayer_Metropolis {
    inherit Module

    constructor {config} {
	set name Metropolis
	set_defaults
    }

    method set_defaults {} {
	global $this-burning
	global $this-monitor
	global $this-thin 
	global $this-kappa
	global $this-use-cvode

	set $this-burning 20000
	set $this-monitor 50000
	set $this-thin    10
	set $this-kappa   0
	set $this-ready 0
	set $this-use-cvode 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	frame $w.f 

	pack $w.f -padx 2 -pady 2 -fill both -expand yes
#	set n "$this-c needexecute "

        iwidgets::entryfield $w.f.burning -labeltext "Burning:" \
	    -validate numeric -command "$this change_burning" 

        iwidgets::entryfield $w.f.monitor -labeltext "Monitor:" \
	    -validate numeric -command "$this change_monitor" 

        iwidgets::entryfield $w.f.thin -labeltext "Thin:" \
	    -validate numeric -command "$this change_thin" 

        iwidgets::entryfield $w.f.kappa -labeltext "Kappa:" \
	    -validate numeric -command "$this change_kappa" 

	checkbutton $w.f.cvode -text CVODE -variable $this-use-cvode

	button $w.f.exec -text "Execute" -command "$this-c exec"

	iwidgets::Labeledframe $w.f.graph -labeltext "Progress"

	pack $w.f.burning $w.f.monitor $w.f.thin $w.f.exec -anchor w
	pack $w.f.cvode -anchor w
	pack $w.f.graph -expand yes -fill both

	$this-c graph-window [$w.f.graph childsite]

	set $this-ready 1
    }

    
    method change_burning {} {
	global $this-burning
	
	set $this-burning [.ui[modname].f.burning get]
	puts "tcl burning [set $this-burning]"
    }
	
    method change_monitor {} {
	global $this-monitor

	set $this-monitor [.ui[modname].f.monitor get]
	puts "tcl monitor [set $this-monitor]"

    }

    method change_thin {} {
	global $this-thin

	set $this-thin [.ui[modname].f.thin get]
	puts "tcl thin [set $this-thin]"
    }

    method change_kappa {} {
	global $this-kappa

	set $this-kappa [.ui[modname].f.kappa get]
	puts "tcl kappa [set $this-kappa]"
    }
}
