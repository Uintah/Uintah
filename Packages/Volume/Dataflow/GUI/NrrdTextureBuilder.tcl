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


catch {rename NrrdTextureBuilder ""}

itcl_class Volume_Visualization_NrrdTextureBuilder {
    inherit Module
    constructor {config} {
	set name NrrdTextureBuilder
	set_defaults
    }
    method set_defaults {} {
	global $this-card_mem
	set $this-card_mem 16
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	set n "$this-c needexecute"
	
	frame $w.f.memframe -relief groove -border 2
	label $w.f.memframe.l -text "Graphics Card Memory (Mb)"
	pack $w.f.memframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.memframe.l -side top -fill x

	frame $w.f.memframe.bf -relief flat -border 2
        set bf $w.f.memframe.bf
	pack $bf -side top -fill x
	radiobutton $bf.b0 -text 4 -variable $this-card_mem -value 4 \
	    -command $n
	radiobutton $bf.b1 -text 8 -variable $this-card_mem -value 8 \
	    -command $n
	radiobutton $bf.b2 -text 16 -variable $this-card_mem -value 16 \
	    -command $n
	radiobutton $bf.b3 -text 32 -variable $this-card_mem -value 32 \
	    -command $n
	radiobutton $bf.b4 -text 64 -variable $this-card_mem -value 64 \
	    -command $n
	radiobutton $bf.b5 -text 128 -variable $this-card_mem -value 128 \
	    -command $n
	radiobutton $bf.b6 -text 256 -variable $this-card_mem -value 256 \
	    -command $n
	pack $bf.b0 $bf.b1 $bf.b2 $bf.b3 $bf.b4 $bf.b5 $bf.b6 -side left -expand yes\
                -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    method set_card_mem {mem} {
	set $this-card_mem $mem
    }
}
