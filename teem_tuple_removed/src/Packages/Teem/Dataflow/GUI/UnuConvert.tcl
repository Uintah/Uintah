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

##
 #  UnuConvert.tcl: The UnuConvert UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Unu_UnuConvert ""}

itcl_class Teem_Unu_UnuConvert {
    inherit Module
    constructor {config} {
        set name UnuConvert
        set_defaults
    }
    method set_defaults {} {
        global $this-type
	set $this-type 5
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-type
	make_labeled_radio $w.f.t "New Type:" "" \
		top $this-type \
		{{char 1} \
		{uchar 2} \
		{short 3} \
		{ushort 4} \
		{int 5} \
		{uint 6} \
		{float 9} \
		{double 10}}
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.t $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
