##
 #  EditPath.tcl
 #
 #  Written by:
 #   David Weinstein & Alexei Samsonov
 #   Department of Computer Science
 #   University of Utah
 #   February 1999, July 2000
 #
 #  Copyright (C) 1999, 2000  SCI Group
 #
 ##

#catch {rename EditPath ""}

itcl_class PSECommon_Salmon_EditPath {
    inherit Module
    protected df ef

    constructor {config} {
        set name EditPath
        set_defaults
    }
    method set_defaults {} {
        global $this-tcl_is_new
	global $this-tcl_rate
	global $this-tcl_curr_view
	global $this-tcl_num_views
	global $this-tcl_intrp_type
	global $this-tcl_speed_mode
	global $this-tcl_acc_mode
	global $this-tcl_is_looped
	global $this-tcl_msg_box
	global $this-tcl_curr_roe
	global $this-tcl_acc_pat
	global $this-tcl_step_size
	global $this-tcl_speed_val
	global $this-tcl_acc_val
	global $this-tcl_msg_box
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
	
	set $this-UI_Init 1

        toplevel $w
	wm title $w "Edit Camera Path"
        wm minsize $w 300 80

        set  df  [frame $w.drv]
	set  ef [frame $w.editor]
	pack $df $ef -side top -fill both -padx 5 -pady 5
	
	frame $ef.fsb
	frame $ef.btn
	frame $ef.info
	frame $ef.sw
	pack  $ef.btn $ef.fsb $ef.info $ef.sw -side top -fill both -pady 3
	
	scale $ef.fsb.fr -variable $this-tcl_curr_view -from 0 -to [set $this-tcl_num_views] \
		-orient horizontal -width 15 -length 150 -command "$this-c get_to_view" -label "Current View:"
	scale $ef.fsb.fsm -variable $this-tcl_step_size -digits 6 -from 0.000001 -to 0.01 \
		-resolution 0.00001 -orient horizontal -width 7 -length 150 -command "$this-c set_step_size" -label "Smoothness:"
	pack $ef.fsb.fr $ef.fsb.fsm -side top -fill x -padx 2

	button $ef.btn.add -text "Add View" -command "$this-c add_vp" -anchor w
	button $ef.btn.del -text "Del View" -command "$this-c rem_vp" -anchor w
	button $ef.btn.ins -text "Ins View" -command "$this-c ins_vp" -anchor w
	button $ef.btn.next -text "Next View" -command "$this-c get_to_view" -anchor w
	pack  $ef.btn.add $ef.btn.del $ef.btn.ins $ef.btn.next -side left -fill both -padx 2

	checkbutton $ef.sw.lp -text "Looped" -variable $this-tcl_is_looped -command "$this-c switch_loop" -anchor w
	checkbutton $ef.sw.bk -text "Reversed" -variable $this-tcl_is_backed -command "$this-c switch_dir" -anchor w
	pack $ef.sw.lp $ef.sw.bk -side top
	
	
	frame $df.ftest
	frame $df.info
	frame $df.modes
	pack  $df.ftest $df.info $df.modes -side top -fill both -pady 3
	
	button $df.ftest.runviews -text "Test Views" -command "$this-c test_views" -anchor w
	button $df.ftest.run -text "Run" -command "$this-c test_path" -anchor w
	button $df.ftest.stop -text "Stop" -command "$this-c stop_test" -anchor w
	button $df.ftest.save -text "Save" -command "$this-c save_path" -anchor w
	
	scale  $df.ftest.sbrate -variable $this-tcl_rate -digits 3 -from 0.01 -to 1 \
		-resolution 0.05 -orient horizontal -width 7 -length 150 -command "$this-c set_rate"
	
	label  $df.ftest.sblabel -text "Rate: " -anchor w
	pack   $df.ftest.runviews $df.ftest.run $df.ftest.stop $df.ftest.save \
	       $df.ftest.sblabel $df.ftest.sbrate -side left -padx 2

	frame $df.info.speed 
	frame $df.info.acc 
	frame $df.info.nstep
	pack $df.info.speed $df.info.acc $df.info.nstep -side top -anchor w -fill x -pady 1
	
	label $df.info.speed.head -text "Speed:\t\t"
	label $df.info.speed.val -textvariable $this-speed_val
	pack $df.info.speed.head $df.info.speed.val -side left -anchor w -padx 2
	
	label $df.info.acc.head -text "Acceleration:\t"
	label $df.info.acc.val -textvariable $this-acc_val
	pack $df.info.acc.head $df.info.acc.val -side left -anchor w -padx 2
	
	label $df.modes.head -text "Modes: "
	radiobutton $df.modes.new -text "New Path" -variable $this-tcl_is_new -value 1 -command "$this-c init_new"
	radiobutton $df.modes.exist -text "Existing Path" -variable $this-tcl_is_new -value 0 -command "$this-c init_exist"
	pack $df.modes.head $df.modes.new $df.modes.exist -side left -padx 2 
    }

    method EraseWarn {t m} {
	set w .ui[modname]
	#set temp [tk_messageBox -title "Modified Path Exist Warning" -parent $w -message "There is modified camera path. Are you sure you want to discard it?" -type okcancel -icon question]
	set temp [tk_messageBox -title $t -parent $w -message $m -type okcancel -icon question]
	case $temp {
	    ok {set $this-tcl_msg_box 1} 
	    cancel {set $this-tcl_msg_box 0}
	}
    }
    
    method refresh {} {
	set w .ui[modname]
	update
	set nv [expr [set $this-tcl_num_views]]
	set len [$w.editor.fsb.fr cget -length]
	if { $nv > 0} {
	    $w.editor.fsb.fr configure -state normal -from 1 -to $nv -sliderlength [expr $len/$nv]
	} else {
	    $w.editor.fsb.fr configure -state disabled -from 0 -to 0 -sliderlength $len
	}
    }
}
