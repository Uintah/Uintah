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
#    File   : NrrdSelectTime.tcl
#    Author : Martin Cole
#    Date   : Tue Sep  9 10:29:51 2003

itcl_class Teem_NrrdData_NrrdSelectTime {
    inherit Module 

    constructor {config} {
        set name NrrdSelectTime

        global $this-selectable_min
        global $this-selectable_max
        global $this-selectable_inc
        global $this-range_min
        global $this-range_max
	global $this-playmode
	global $this-current
	global $this-execmode
	global $this-delay
	global $this-inc-amount


        set_defaults
    }

    method set_defaults {} {    
        set $this-selectable_min     0
        set $this-selectable_max     100
        set $this-selectable_inc     1
        set $this-range_min          0
        set $this-range_max          0
	set $this-playmode           once
	set $this-current            0
	set $this-execmode           "init"
	set $this-delay              0
	set $this-inc-amount        1
    }

    method run_update {} {
	set $this-execmode "update"
	$this-c needexecute
    }

    method run_step {} {
	set $this-execmode "step"
	$this-c needexecute
    }

    method run_play {} {
	set $this-execmode "play"
	$this-c needexecute
    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	
	frame $w.loc -borderwidth 2
	frame $w.playmode -relief groove -borderwidth 2
	frame $w.execmode -relief groove -borderwidth 2

        scale $w.loc.min -variable $this-range_min -label "Start " \
		-showvalue true -orient horizontal -relief groove -length 200
        scale $w.loc.max -variable $this-range_max -label "End " \
		-showvalue true -orient horizontal -relief groove -length 200

	frame $w.loc.e
	frame $w.loc.e.l
	frame $w.loc.e.r

	label $w.loc.e.l.curlabel -text "Current Value" -just left
	entry $w.loc.e.r.curentry -width 10 -textvariable $this-current

	label $w.loc.e.l.inclabel -text "Increment" -justify left
	entry $w.loc.e.r.incentry -width 8 -textvariable $this-inc-amount

	pack $w.loc.e.l.curlabel $w.loc.e.l.inclabel \
	      -side top -anchor w

	pack $w.loc.e.r.curentry $w.loc.e.r.incentry \
              -side top -anchor w

	pack $w.loc.e.l $w.loc.e.r -side left -expand yes -fill both

        pack $w.loc.min $w.loc.max $w.loc.e \
	    -side top -expand yes -fill both -pady 2


	label $w.playmode.label -text "Play Mode"
	radiobutton $w.playmode.once -text "Once" \
		-variable $this-playmode -value once
	radiobutton $w.playmode.loop -text "Loop" \
		-variable $this-playmode -value loop
	radiobutton $w.playmode.bounce1 -text "Bounce1" \
		-variable $this-playmode -value bounce1
	radiobutton $w.playmode.bounce2 -text "Bounce2" \
		-variable $this-playmode -value bounce2

	radiobutton $w.playmode.inc_w_exec -text "Increment with Execute" \
	    -variable $this-playmode -value inc_w_exec

	frame $w.playmode.delay
	label $w.playmode.delay.label -text "Delay (ms)" \
		-width 10 -just left
	entry $w.playmode.delay.entry -width 10 -textvariable $this-delay
	pack $w.playmode.delay.label $w.playmode.delay.entry \
		-side left -anchor n -expand yes -fill x


	pack $w.playmode.label -side top -expand yes -fill both
	pack $w.playmode.once $w.playmode.loop \
	    $w.playmode.bounce1 $w.playmode.bounce2 $w.playmode.inc_w_exec\
	    $w.playmode.delay -side top -anchor w


        button $w.execmode.play -text "Play" -command "$this run_play"
        button $w.execmode.stop -text "Stop" -command "$this-c stop"
        button $w.execmode.step -text "Step" -command "$this run_step"
        pack $w.execmode.play $w.execmode.stop $w.execmode.step \
		-side left -fill both -expand yes

        pack $w.loc $w.playmode $w.execmode \
		-padx 5 -pady 5 -fill both -expand yes

	update
    }

    method update {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            #puts "updating!"

            $w.loc.min config -from [set $this-selectable_min]
            $w.loc.min config -to   [set $this-selectable_max]
            $w.loc.min config -label "Start"
            $w.loc.max config -from [set $this-selectable_min]
            $w.loc.max config -to   [set $this-selectable_max]
            $w.loc.max config -label "End"

	    #pack $w.loc.r $w.loc.c $w.loc.ff $w.loc.b -side top -fill x -expand yes
        }
    }
}
