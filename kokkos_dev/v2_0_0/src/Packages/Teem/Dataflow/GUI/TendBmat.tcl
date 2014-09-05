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
#    File   : TendBmat.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendBmat ""}

itcl_class Teem_Tend_TendBmat {
    inherit Module
    constructor {config} {
        set name TendBmat
        set_defaults
    }
    method set_defaults {} {
        global $this-gradient_list
        set $this-gradient_list ""


    }

    method update_text {} {
	set w .ui[modname]
	set $this-gradient_list [$w.f.options.gradient_list get 1.0 end]
    }

    method send_text {} {
	$this update_text
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes



	option add *textBackground white	
	iwidgets::scrolledtext $w.f.options.gradient_list -vscrollmode dynamic \
		-labeltext "List of gradients. example: (one gradient per line) 0.5645 0.32324 0.4432454"
	set cmmd "$this send_text"
	bind $w.f.options.gradient_list <Leave> "$this update_text"
	catch {$w.f.options.gradient_list insert end [set $this-gradient_list]}

        pack $w.f.options.gradient_list -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
