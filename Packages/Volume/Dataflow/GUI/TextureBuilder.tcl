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


catch {rename TextureBuilder ""}

itcl_class Volume_Visualization_TextureBuilder {
    inherit Module
    constructor {config} {
	set name TextureBuilder
	set_defaults
    }
    method set_defaults {} {
	global $this-card_mem
	global $this-min
	global $this-max
	global $this-is_fixed
	set $this-min 0
	set $this-max 1
	set $this-is_fixed 0
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

	global $this-is_fixed
        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Scale"  -variable $this-is_fixed \
		-value 0 -command "$this autoScale"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Scale"  -variable $this-is_fixed \
		-value 1 -command "$this fixedScale"
        pack $w.f2.b -side left

        frame $w.f3 -relief flat
        pack $w.f3 -side top -expand yes -fill x
        
        label $w.f3.l1 -text "min:  "
        entry $w.f3.e1 -textvariable $this-min

        label $w.f3.l2 -text "max:  "
        entry $w.f3.e2 -textvariable $this-max
        pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
            -expand yes -fill x -padx 2 -pady 2

        bind $w.f3.e1 <Return> $n
        bind $w.f3.e2 <Return> $n

       if { [set $this-is_fixed] } {
            $w.f2.b select
            $this fixedScale
        } else {
            $w.f1.b select
            $this autoScale
        }

	makeSciButtonPanel $w $w $this
	moveToCursor $w

    }

    method autoScale { } {
        global $this-is_fixed
        set w .ui[modname]
        
        set $this-is_fixed 0

        set color "#505050"

        $w.f3.l1 configure -foreground $color
        $w.f3.e1 configure -state disabled -foreground $color
        $w.f3.l2 configure -foreground $color
        $w.f3.e2 configure -state disabled -foreground $color


   }	

    method fixedScale { } {
        global $this-is_fixed
        set w .ui[modname]

        set $this-is_fixed 1


        $w.f3.l1 configure -foreground black
        $w.f3.e1 configure -state normal -foreground black
        $w.f3.l2 configure -foreground black
        $w.f3.e2 configure -state normal -foreground black
        
    }

}
