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


catch {rename PPexample ""}

package require Iwidgets 3.0

itcl_class MIT_Test_PPexample {
    inherit Module

    variable index

    constructor {config} {
	set name PPexample
    }

    method ui { args } {

	puts "args = $args"
	set index 0
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	frame $w.f 
	pack $w.f

	$this-c set-window $w.f
    }

}


