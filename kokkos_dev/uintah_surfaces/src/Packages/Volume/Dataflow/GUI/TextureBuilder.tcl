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
	global $this-card_mem_auto
	global $this-vmin
	global $this-vmax
	global $this-gmin
	global $this-gmax
	global $this-is_fixed
	set $this-vmin 0
	set $this-vmax 1
	set $this-gmin 0
	set $this-gmax 1
	set $this-is_fixed 0
	set $this-card_mem 16
	set $this-card_mem_auto 1
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
	set s "$this state"
	
	frame $w.f.memframe -relief groove -border 2
	label $w.f.memframe.l -text "Graphics Card Memory (MB)"
	pack $w.f.memframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.memframe.l -side top -fill x
	checkbutton $w.f.memframe.auto -text "Autodetect" -relief flat \
            -variable $this-card_mem_auto -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
	pack $w.f.memframe.auto -side top -fill x

	frame $w.f.memframe.bf -relief flat -border 2
        set bf $w.f.memframe.bf
	pack $bf -side top -fill x
	radiobutton $bf.b0 -text 4 -variable $this-card_mem -value 4 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b1 -text 8 -variable $this-card_mem -value 8 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b2 -text 16 -variable $this-card_mem -value 16 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b3 -text 32 -variable $this-card_mem -value 32 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b4 -text 64 -variable $this-card_mem -value 64 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b5 -text 128 -variable $this-card_mem -value 128 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b6 -text 256 -variable $this-card_mem -value 256 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b7 -text 512 -variable $this-card_mem -value 512 \
	    -command $n -state disabled -foreground darkgrey
	pack $bf.b0 $bf.b1 $bf.b2 $bf.b3 $bf.b4 $bf.b5 $bf.b6 $bf.b7 -side left -expand yes\
                -fill x

	global $this-is_fixedmin
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
        label $w.f3.l1 -text "value min:  "
        entry $w.f3.e1 -textvariable $this-vmin
        label $w.f3.l2 -text "value max:  "
        entry $w.f3.e2 -textvariable $this-vmax
        pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
            -expand yes -fill x -padx 2 -pady 2

        frame $w.f4 -relief flat
        pack $w.f4 -side top -expand yes -fill x
        label $w.f4.l1 -text " grad min:  "
        entry $w.f4.e1 -textvariable $this-gmin
        label $w.f4.l2 -text " grad max:  "
        entry $w.f4.e2 -textvariable $this-gmax
        pack $w.f4.l1 $w.f4.e1 $w.f4.l2 $w.f4.e2 -side left \
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

        $w.f4.l1 configure -foreground $color
        $w.f4.e1 configure -state disabled -foreground $color
        $w.f4.l2 configure -foreground $color
        $w.f4.e2 configure -state disabled -foreground $color
   }	

    method fixedScale { } {
        global $this-is_fixed
        set w .ui[modname]

        set $this-is_fixed 1


        $w.f3.l1 configure -foreground black
        $w.f3.e1 configure -state normal -foreground black
        $w.f3.l2 configure -foreground black
        $w.f3.e2 configure -state normal -foreground black

        $w.f4.l1 configure -foreground black
        $w.f4.e1 configure -state normal -foreground black
        $w.f4.l2 configure -foreground black
        $w.f4.e2 configure -state normal -foreground black
        
    }

    method set_card_mem {mem} {
	set $this-card_mem $mem
    }
    method set_card_mem_auto {auto} {
	set $this-card_mem_auto $auto
    }
    method state {} {
	set w .ui[modname]
	if {[winfo exists $w] == 0} {
	    return
	}
	if {[set $this-card_mem_auto] == 1} {
            $this deactivate $w.f.memframe.bf.b0
            $this deactivate $w.f.memframe.bf.b1
            $this deactivate $w.f.memframe.bf.b2
            $this deactivate $w.f.memframe.bf.b3
            $this deactivate $w.f.memframe.bf.b4
            $this deactivate $w.f.memframe.bf.b5
            $this deactivate $w.f.memframe.bf.b6
            $this deactivate $w.f.memframe.bf.b7
	} else {
            $this activate $w.f.memframe.bf.b0
            $this activate $w.f.memframe.bf.b1
            $this activate $w.f.memframe.bf.b2
            $this activate $w.f.memframe.bf.b3
            $this activate $w.f.memframe.bf.b4
            $this activate $w.f.memframe.bf.b5
            $this activate $w.f.memframe.bf.b6
            $this activate $w.f.memframe.bf.b7
        }
    }
    method activate { w } {
	$w configure -state normal -foreground black
    }
    method deactivate { w } {
	$w configure -state disabled -foreground darkgrey
    }
}
