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

catch {rename BioPSE_Inverse_DipoleSearch ""}

itcl_class BioPSE_Inverse_DipoleSearch {
    inherit Module

    constructor {config} {
	set name DipoleSearch
	set_defaults
    }

    method set_defaults {} {	
	global $this-use_cache_gui_
	set $this-use_cache_gui_ 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
        toplevel $w
	wm minsize $w 100 50
	
	set n "$this-c needexecute "
        
	frame $w.g
        button $w.g.go -text "Execute" -relief raised -command $n 
        button $w.g.p -text "Pause" -relief raised -command "$this-c pause"
        button $w.g.np -text "Unpause" -relief raised -command "$this-c unpause"
	button $w.g.stop -text "Stop" -relief raised -command "$this-c stop"
	pack $w.g.go $w.g.p $w.g.np $w.g.stop -side left -fill x
	global $this-use_cache_gui_
	checkbutton $w.b -text "UseCache" -variable $this-use_cache_gui_
	pack $w.g $w.b -side top -fill x
    }
}

