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

itcl_class SCIRun_Math_MatrixSelectVector {
    inherit Module 

    constructor {config} {
        set name MatrixSelectVector

        global $this-row_or_col
        global $this-selectable_min
        global $this-selectable_max
        global $this-selectable_inc
	global $this-selectable_units
        global $this-range_min
        global $this-range_max
	global $this-playmode
	global $this-current
	global $this-execmode
	global $this-delay
	global $this-inc-amount
	global $this-send-amount


        set_defaults
    }

    method set_defaults {} {    
        set $this-row_or_col         row
        set $this-selectable_min     0
        set $this-selectable_max     100
        set $this-selectable_inc     1
	set $this-selectable_units   ""
        set $this-range_min          0
        set $this-range_max          0
	set $this-playmode           once
	set $this-current            0
	set $this-execmode           "init"
	set $this-delay              0
	set $this-inc-amount        1
	set $this-send-amount       1
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

	
	frame $w.location -borderwidth 2
	frame $w.playmode -relief groove -borderwidth 2
	frame $w.execmode -relief groove -borderwidth 2

	frame $w.location.roc
        radiobutton $w.location.roc.row -text "Row" \
		-variable $this-row_or_col \
		-value row -command "$this run_update"
        radiobutton $w.location.roc.col -text "Column" \
		-variable $this-row_or_col \
		-value col -command "$this run_update"
	pack $w.location.roc.row $w.location.roc.col \
		-side left -expand yes -fill both

        scale $w.location.min -variable $this-range_min -label "Start " \
		-showvalue true -orient horizontal -relief groove -length 200
        scale $w.location.max -variable $this-range_max -label "End " \
		-showvalue true -orient horizontal -relief groove -length 200

	frame $w.location.cur
	label $w.location.cur.label -text "Current Value" -width 10 -just left
	entry $w.location.cur.entry -width 10 -textvariable $this-current
	pack $w.location.cur.label $w.location.cur.entry \
		-side left -anchor n -expand yes -fill x

        pack $w.location.roc $w.location.min $w.location.max $w.location.cur \
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

	frame $w.playmode.delay
	label $w.playmode.delay.label -text "Delay (ms)" \
		-width 10 -just left
	entry $w.playmode.delay.entry -width 10 -textvariable $this-delay
	pack $w.playmode.delay.label $w.playmode.delay.entry \
		-side left -anchor n -expand yes -fill x

	frame $w.playmode.inc
	label $w.playmode.inc.label -text "Increment" -width 16 -justify left
	entry $w.playmode.inc.entry -width 8 -textvariable $this-inc-amount
	pack $w.playmode.inc.label $w.playmode.inc.entry \
		-side left -anchor n -expand yes -fill x

	frame $w.playmode.send
	label $w.playmode.send.label -text "Amount to Send" \
	    -width 16 -justify left
	entry $w.playmode.send.entry -width 8 -textvariable $this-send-amount
	pack $w.playmode.send.label $w.playmode.send.entry \
		-side left -anchor n -expand yes -fill x

	pack $w.playmode.label -side top -expand yes -fill both
	pack $w.playmode.once $w.playmode.loop \
		$w.playmode.bounce1 $w.playmode.bounce2 $w.playmode.delay \
	        $w.playmode.inc $w.playmode.send -side top -anchor w


        button $w.execmode.play -text "Play" -command "$this run_play"
        button $w.execmode.stop -text "Stop" -command "$this-c stop"
        button $w.execmode.step -text "Step" -command "$this run_step"
        pack $w.execmode.play $w.execmode.stop $w.execmode.step \
		-side left -fill both -expand yes

        pack $w.location $w.playmode $w.execmode \
		-padx 5 -pady 5 -fill both -expand yes

	update
    }

    method update {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            #puts "updating!"

            $w.location.min config -from [set $this-selectable_min]
            $w.location.min config -to   [set $this-selectable_max]
            $w.location.min config -label [concat "Start " [set $this-selectable_units]]
            $w.location.max config -from [set $this-selectable_min]
            $w.location.max config -to   [set $this-selectable_max]
            $w.location.max config -label [concat "End " [set $this-selectable_units]]

	    #pack $w.location.r $w.location.c $w.location.ff $w.location.b -side top -fill x -expand yes
        }
    }
}
