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

#catch {rename Viewer ""} 

itcl::class SCIRun_Render_Viewer {
    inherit ModuleGui
    
    protected variable nextrid 0
    
    public variable make_progress_graph 0
    public variable make_time 0
    public variable viewwindow ""
    
    constructor {} {
	set name Viewer
	global global_viewer
	set global_viewer $this
    }
    
    destructor {
	foreach rid $viewwindow {
	    destroy .ui[$rid modname]
	    
	    $rid delete
	}
    }
    
    method makeViewWindowID {} {
	set id $this-ViewWindow_$nextrid
	incr nextrid
	while {[::info commands $id] != ""} {
	    set id $this-ViewWindow_$nextrid
	    incr nextrid
	}
	return $id
    }
    
    method ui {{rid -1}} {
	if {$rid == -1} {
	    set rid [makeViewWindowID]
	}
	$this-c addviewwindow $rid
	lappend viewwindow $rid
    }
}

