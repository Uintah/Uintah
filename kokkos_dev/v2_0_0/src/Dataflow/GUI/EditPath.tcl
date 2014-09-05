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
 #  EditPath.tcl
 #  Written by:
 #   David Weinstein & Alexei Samsonov
 #   Department of Computer Science
 #   University of Utah
 #   February 1999, July 2000
 #  Copyright (C) 1999, 2000  SCI Group
 ##
itcl_class SCIRun_Render_EditPath {
    inherit Module
   
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
	global $this-tcl_acc_mode
	global $this-tcl_is_looped
	global $this-tcl_msg_box
	global $this-tcl_curr_viewwindow
	global $this-tcl_step_size
	global $this-tcl_speed_val
	global $this-tcl_acc_val
	global $this-tcl_stop
	global $this-tcl_widget_show

        set $this-tcl_is_new 1
	set $this-tcl_rate 1
	set $this-tcl_curr_view 0
	set $this-tcl_num_views 0
	set $this-tcl_intrp_type 2
	set $this-tcl_acc_mode 1
	set $this-tcl_is_looped 0
	set $this-tcl_msg_box 0
	set $this-tcl_curr_viewwindow 0
	set $this-tcl_step_size 0.01
	set $this-tcl_speed_val 0
	set $this-tcl_acc_val 0
	set $this-tcl_stop 0
	set $this-tcl_widget_show 0
	
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
	wm title $w "Edit Camera Path"
        wm minsize $w 300 80

        set  df  [frame $w.drv -relief groove]
	set  ef  [frame $w.editor -relief groove]
	pack $df $ef -side top -fill both -padx 5 -pady 5
	
	#**************************************************************
	# editor frame

	frame $ef.fsb
	frame $ef.btn
	frame $ef.sw
	frame $ef.mkc
	pack  $ef.btn $ef.fsb $ef.mkc $ef.sw -side top -fill both -pady 3
	scale $ef.fsb.fr -variable $this-tcl_curr_view -from 0 -to [set $this-tcl_num_views] \
		-orient horizontal -width 15 -command "$this-c get_to_view" -label "Current View:"
	scale $ef.fsb.fsm -variable $this-tcl_step_size -digits 6 -from 0.0001 -to 0.1 \
		-resolution 0.0001 -orient horizontal -width 7 -label "Path Step:"
	scale $ef.fsb.sp -variable $this-tcl_speed_val -digits 4 -from 0.1 -to 10 \
		-resolution 0.01 -orient horizontal -width 7 -label "Speed:"
	pack $ef.fsb.fr $ef.fsb.fsm $ef.fsb.sp -side top -fill x -padx 2

	button $ef.btn.add -text "Add View" -command "$this-c add_vp" -anchor w
	button $ef.btn.del -text "Del View" -command "$this-c rem_vp" -anchor w
	button $ef.btn.ins -text "Ins View" -command "$this-c ins_vp" -anchor w
	button $ef.btn.rpl -text "Rpl View" -command "$this-c rpl_vp" -anchor w
	button $ef.btn.prev -text "Prev View" -command "$this-c prev_view" -anchor w
	button $ef.btn.next -text "Next View" -command "$this-c next_view" -anchor w	
	pack  $ef.btn.add $ef.btn.del $ef.btn.ins $ef.btn.rpl $ef.btn.prev $ef.btn.next -side left -fill both -padx 2

	button $ef.mkc.make -text "Make Circle" -command "$this-c mk_circle_path" -anchor w
	checkbutton $ef.mkc.ws -text "Center-Widget" -command "$this-c w_show" -variable $this-tcl_widg_show -onvalue 1 -offvalue 0 -padx 5
	pack $ef.mkc.make $ef.mkc.ws -side left -fill both -padx 4 -pady 4
	
	frame $ef.sw.int -relief ridge
	frame $ef.sw.spd -relief ridge
	frame $ef.sw.info -relief ridge
	pack $ef.sw.int $ef.sw.spd $ef.sw.info -side left -padx 10 -anchor n

	
	label $ef.sw.spd.lb -text "Acceleration:" -anchor w
	radiobutton $ef.sw.spd.sm -text "Smooth Start/End" -variable $this-tcl_acc_mode  -anchor w -value 1 
	radiobutton $ef.sw.spd.no -text "No acceleration " -variable $this-tcl_acc_mode  -anchor w -value 0
	radiobutton $ef.sw.spd.us -text "User Speeds Only" -variable $this-tcl_acc_mode  -anchor w -value 2
	pack $ef.sw.spd.lb $ef.sw.spd.no $ef.sw.spd.sm $ef.sw.spd.us -side top -pady 3 -padx 2 -anchor w
	
	label $ef.sw.int.lb -text "Interpolation Type:" -anchor w
	radiobutton $ef.sw.int.lin -text "Linear" -variable $this-tcl_intrp_type  -anchor w -value 1
	radiobutton $ef.sw.int.kf -text "None" -variable $this-tcl_intrp_type  -anchor w -value 0
	radiobutton $ef.sw.int.cub -text "Cubic" -variable $this-tcl_intrp_type  -anchor w -value 2
	pack $ef.sw.int.lb $ef.sw.int.kf $ef.sw.int.lin $ef.sw.int.cub -side top -pady 3 -padx 2 -anchor w
	

	#***************************************************************
	# driver frame
	frame $df.ftest
	frame $df.info -relief sunken
	frame $df.modes
	frame $df.sw
	frame $df.out
	pack  $df.ftest $df.info $df.modes $df.out $df.sw -side top -fill both -pady 3 -padx 3

#	button $df.ftest.runviews -text "Test Views" -command "$this-c test_views" -anchor w
	button $df.ftest.run -text "Run" -command "$this-c test_path" -anchor w
	button $df.ftest.stop -text "Stop" -command "set $this-tcl_stop 1"  -anchor w
	button $df.ftest.save -text "Save" -command "$this-c save_path" -anchor w
	
	scale  $df.ftest.sbrate -variable $this-tcl_rate -digits 4 -from 0.05 -to 25 \
		-resolution 0.05 -orient horizontal -width 7 -length 300
	
	label  $df.ftest.sblabel -text "Rate: " -anchor s
	pack   $df.ftest.run $df.ftest.stop $df.ftest.save \
	       $df.ftest.sblabel $df.ftest.sbrate -side left -padx 2

	frame $df.info.speed 
	frame $df.info.acc
	frame $df.info.nv
	frame $df.info.msg -relief raised
	pack $df.info.speed $df.info.acc  $df.info.nv $df.info.msg -side top -anchor w -fill x -pady 1
	
	#label $df.info.speed.head -text "Speed:\t"
	#label $df.info.speed.val -textvariable $this-tcl_speed_val
	#pack $df.info.speed.head $df.info.speed.val -side left -anchor w -padx 2
	
	label $df.info.acc.head -text "Acceleration:\t"
	label $df.info.acc.val -textvariable $this-tcl_acc_val
	pack $df.info.acc.head $df.info.acc.val -side left -anchor w -padx 2
	
	label $df.info.nv.a -text "# of key frames:"  -anchor w
	label $df.info.nv.b -textvariable $this-tcl_num_views  -anchor w
	pack $df.info.nv.a $df.info.nv.b -side left -padx 2

	label $df.info.msg.h -text "--\t" -anchor w
	label $df.info.msg.b -textvariable $this-tcl_info -anchor w
	pack $df.info.msg.h $df.info.msg.b -side left -padx 2 -pady 2

	label $df.modes.head -text "Modes: "
	radiobutton $df.modes.new -text "New Path" -variable $this-tcl_is_new -value 1 -command "$this-c init_new"
	radiobutton $df.modes.exist -text "Existing Path" -variable $this-tcl_is_new -value 0 -command "$this-c init_exist"
	pack $df.modes.head $df.modes.new $df.modes.exist -side left -padx 2
	
	label $df.out.head -text "Output: "
	radiobutton $df.out.ogeom -text "Geometry Port" -variable $this-tcl_send_dir -value 0
	radiobutton $df.out.oview -text "CameraView Port" -variable $this-tcl_send_dir -value 1
	pack  $df.out.head $df.out.ogeom  $df.out.oview -side left -padx 2
	
	checkbutton $df.sw.lp -text "Looped" -variable $this-tcl_is_looped  -anchor w
	checkbutton $df.sw.bk -text "Reversed" -variable $this-tcl_is_backed  -anchor w
	pack $df.sw.lp $df.sw.bk -side left -anchor w
	
	refresh
	set $this-UI_Init 1
    }

    method EraseWarn {t m} {
	set w .ui[modname]

	set temp [tk_messageBox -title $t -parent $w -message $m -type okcancel -icon question]
	case $temp {
	    ok {set $this-tcl_msg_box 1}
	    cancel {set $this-tcl_msg_box 0}
	}
    }
    
    method refresh {} {
	update
	set w .ui[modname]
	set nv [expr [set $this-tcl_num_views]]
	set len [$w.editor.fsb.fr cget -length]
	if { $nv > 0} {
	    $w.editor.fsb.fr configure -state normal -from 0 -to [expr $nv-1] -showvalue 1 
	} else {
	    $w.editor.fsb.fr configure -state disabled -from 0 -to 0  -showvalue 0
	}
	update_tcl
    }

    method update_tcl {} {
	set w .ui[modname]
	set df $w.drv
	set ef $w.editor
	
	# brute force fixes of GUI update problem ( didn't figure it out yet)
	
	$df.modes.new configure -variable $this-tcl_is_new
	$df.modes.exist configure -variable $this-tcl_is_new
	$ef.sw.int.lin configure -variable $this-tcl_intrp_type
	$ef.sw.int.kf configure -variable $this-tcl_intrp_type
	$ef.sw.int.cub configure -variable $this-tcl_intrp_type
	$ef.sw.spd.sm configure -variable $this-tcl_acc_mode
	$ef.sw.spd.no configure -variable $this-tcl_acc_mode
	$ef.sw.spd.us configure -variable $this-tcl_acc_mode
	$df.out.ogeom configure -variable $this-tcl_send_dir
	$df.out.oview configure -variable $this-tcl_send_dir
	$ef.fsb.sp configure -variable $this-tcl_speed_val
	update
    }
}


