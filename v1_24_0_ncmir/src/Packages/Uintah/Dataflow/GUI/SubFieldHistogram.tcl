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


catch {rename SubFieldHistogram ""}

itcl_class Uintah_Visualization_SubFieldHistogram {
    inherit Module
    constructor {config} {
	set name SubFieldHistogram
	set_defaults
    }
    method set_defaults {} {
	global $this-min_
	global $this-max_
	global $this-is_fixed_
	set $this-min_ 0
	set $this-max_ 1
	set $this-is_fixed_ 0
    }


    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
#	wm minsize $w 250 300
	set n "$this-c needexecute "

	global $this-is_fixed_
        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Scale"  -variable $this-is_fixed_ \
		-value 0 -command "$this autoScale"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Scale"  -variable $this-is_fixed_ \
		-value 1 -command "$this fixedScale"
        pack $w.f2.b -side left

        frame $w.f3 -relief flat
        pack $w.f3 -side top -expand yes -fill x
        
        label $w.f3.l1 -text "min:  "
        entry $w.f3.e1 -textvariable $this-min_

        label $w.f3.l2 -text "max:  "
        entry $w.f3.e2 -textvariable $this-max_
        pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
            -expand yes -fill x -padx 2 -pady 2

        bind $w.f3.e1 <Return> $n
        bind $w.f3.e2 <Return> $n


	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x

	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x

       if { [set $this-is_fixed_] } {
            $w.f2.b select
            $this fixedScale
        } else {
            $w.f1.b select
	    global $this-is_fixed_
	    set w .ui[modname]
	    
	    set $this-is_fixed_ 0
	    
	    set color "#505050"
	    
	    $w.f3.l1 configure -foreground $color
	    $w.f3.e1 configure -state disabled -foreground $color
	    $w.f3.l2 configure -foreground $color
	    $w.f3.e2 configure -state disabled -foreground $color
        }
    }
    method autoScale { } {
        global $this-is_fixed_
        set w .ui[modname]
        
        set $this-is_fixed_ 0

        set color "#505050"

        $w.f3.l1 configure -foreground $color
        $w.f3.e1 configure -state disabled -foreground $color
        $w.f3.l2 configure -foreground $color
        $w.f3.e2 configure -state disabled -foreground $color
	$this-c needexecute	

    }

    method fixedScale { } {
        global $this-is_fixed_
        set w .ui[modname]

        set $this-is_fixed_ 1


        $w.f3.l1 configure -foreground black
        $w.f3.e1 configure -state normal -foreground black
        $w.f3.l2 configure -foreground black
        $w.f3.e2 configure -state normal -foreground black
        
    }

}
