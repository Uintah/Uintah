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

#  DebugSettings.tcl
#  Written by:
#   James T. Purciful
#   Department of Computer Science
#   University of Utah
#   Oct. 1994
#  Copyright (C) 1994 SCI Group

proc showDebugSettings {} {
    global debugsettings_window
    if [catch {raise $debugsettings_window}] {
	toplevel .dsw
	wm title .dsw "Debug Settings"
	wm iconname .dsw "DebugSettings"
	wm minsize .dsw 100 100
	set DebugSettings_window .dsw

	canvas .dsw.canvas -yscrollcommand ".dsw.scroll set" -borderwidth 0 \
		-scrollregion {0c 0c 8c 50c} \
		-width 8c -height 8c
	pack .dsw.canvas -side left -padx 2 -pady 2 -fill both -expand yes

	frame .dsw.canvas.frame -borderwidth 2
	frame .dsw.canvas.frame.l
	pack .dsw.canvas.frame.l -side left -anchor w -padx 4
	frame .dsw.canvas.frame.r
	pack .dsw.canvas.frame.r -side right -anchor w
	pack .dsw.canvas.frame
	.dsw.canvas create window 0 0 -window .dsw.canvas.frame -anchor nw

	scrollbar .dsw.scroll -relief sunken -command ".dsw.canvas yview"
	pack .dsw.scroll -fill y -side right -padx 2 -pady 2

	set ds [debugsettings]
	
	for {set i 0} {$i < [llength $ds]} {incr i 1} {
	    set dbg [lindex $ds $i]
	    set module [lindex $dbg 0]
	    set items [lindex $dbg 1]
	    frame .dsw.canvas.frame.l.$i
	    label .dsw.canvas.frame.l.$i.module -text $module
	    pack .dsw.canvas.frame.l.$i.module -side left -padx 2 -pady 2 -anchor w
	    frame .dsw.canvas.frame.r.$i
	    for {set j 0} {$j < [llength $items]} {incr j 1} {
		set item [lindex $items $j]
		checkbutton .dsw.canvas.frame.r.$i.$j -text $item \
			-variable $item-$module -relief groove
		pack .dsw.canvas.frame.r.$i.$j -side left -padx 2 -pady 2 -anchor w
	    }
	    pack .dsw.canvas.frame.l.$i -anchor w
	    pack .dsw.canvas.frame.r.$i -anchor w
	}
    }
    
    #  proc updateDebugSettings {lineheight tleft gleft gwidth font old_ntasks}
    if {[winfo exists .dsw] == 0} {
	return
    }
}
