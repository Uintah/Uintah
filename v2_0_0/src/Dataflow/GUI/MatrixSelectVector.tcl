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
	global $this-dependence
	global $this-current
	global $this-execmode
	global $this-delay
	global $this-inc-amount
	global $this-send-amount
	global $this-update-function

        set_defaults
    }

    method set_defaults {} {    
        set $this-row_or_col         row
        set $this-selectable_min     0
        set $this-selectable_max     100
        set $this-selectable_inc     1
	set $this-selectable_units   ""
        set $this-range_min          0
        set $this-range_max          100
	set $this-playmode           once
	set $this-dependence         independent
	set $this-current            0
	set $this-execmode           "init"
	set $this-delay              0
	set $this-inc-amount        1
	set $this-send-amount       1
	set $this-update-function   1
	trace variable $this-update-function w "$this update_wrapper"
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

	# Save range, creating the scale resets it to defaults.
	set rmin [set $this-range_min]
	set rmax [set $this-range_max]

	frame $w.loc -borderwidth 2
	frame $w.playmode -relief groove -borderwidth 2
	frame $w.dependence -relief groove -borderwidth 2
	frame $w.execmode -relief groove -borderwidth 2

	frame $w.loc.roc
        radiobutton $w.loc.roc.row -text "Row" \
		-variable $this-row_or_col \
		-value row -command "$this run_update"
        radiobutton $w.loc.roc.col -text "Column" \
		-variable $this-row_or_col \
		-value col -command "$this run_update"
	pack $w.loc.roc.row $w.loc.roc.col \
		-side left -expand yes -fill both

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

	label $w.loc.e.l.sendlabel -text "Amount to Send" -justify left
	entry $w.loc.e.r.sendentry -width 8 -textvariable $this-send-amount

	pack $w.loc.e.l.curlabel $w.loc.e.l.inclabel $w.loc.e.l.sendlabel \
	      -side top -anchor w

	pack $w.loc.e.r.curentry $w.loc.e.r.incentry $w.loc.e.r.sendentry \
              -side top -anchor w

	pack $w.loc.e.l $w.loc.e.r -side left -expand yes -fill both

        pack $w.loc.roc $w.loc.min $w.loc.max $w.loc.e \
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


	label $w.dependence.label -text "Downstream Dependencies"
	radiobutton $w.dependence.independent -text \
	        "Column and Index are Independent" \
	        -variable $this-dependence -value independent
	radiobutton $w.dependence.dependent -text \
	        "Column and Index are Dependent" \
		-variable $this-dependence -value dependent
	pack $w.dependence.label -side top -expand yes -fill both
	pack $w.dependence.independent $w.dependence.dependent \
	        -side top -anchor w


	frame $w.playmode.delay
	label $w.playmode.delay.label -text "Delay (ms)" \
		-width 10 -just left
	entry $w.playmode.delay.entry -width 10 -textvariable $this-delay
	pack $w.playmode.delay.label $w.playmode.delay.entry \
		-side left -anchor n -expand yes -fill x


	pack $w.playmode.label -side top -expand yes -fill both
	pack $w.playmode.once $w.playmode.loop \
		$w.playmode.bounce1 $w.playmode.bounce2 $w.playmode.delay \
	        -side top -anchor w


        button $w.execmode.play -text "Play" -command "$this run_play"
        button $w.execmode.stop -text "Stop" -command "$this-c stop"
        button $w.execmode.step -text "Step" -command "$this run_step"
        pack $w.execmode.play $w.execmode.stop $w.execmode.step \
		-side left -fill both -expand yes

        pack $w.loc $w.playmode $w.dependence $w.execmode \
		-padx 5 -pady 5 -fill both -expand yes

	update

	# Restore range to pre-loaded value
	set $this-range_min $rmin
	set $this-range_max $rmax
    }

    method update {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            $w.loc.min config -from [set $this-selectable_min]
            $w.loc.min config -to   [set $this-selectable_max]
            $w.loc.min config -label [concat "Start " [set $this-selectable_units]]
            $w.loc.max config -from [set $this-selectable_min]
            $w.loc.max config -to   [set $this-selectable_max]
            $w.loc.max config -label [concat "End " [set $this-selectable_units]]
        }
    }

    method update_wrapper {name element op} {
	update
    }
}
