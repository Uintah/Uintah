# ScalarFieldProbe.tcl
# by Alexei Samsonov
# June 2000

itcl_class PSECommon_Fields_ScalarFieldProbe {
    inherit Module
    protected IsInit 0
    constructor {config} {
	set name ScalarFieldProbe
	set_defaults
    }
    method set_defaults {} {
	global $this-minVal
	global $this-maxVal
	global $this-gridType
	global $this-w_x
	global $this-w_y
	global $this-w_z
	global $this-bnd1
	global $this-bnd2
	global $this-num_elements
    }

    method ui {} {
        set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	set IsInit 1
	text $w.text -relief raised -bd 2 -height 12 -width 65 
	update_field

        frame $w.widg -relief flat -borderwidth 2 
	label $w.widg.wpos -text "Widget Coordinates:\t"
	label $w.widg.wx
	label $w.widg.wy
	label $w.widg.wz
	pack $w.widg.wpos $w.widg.wx $w.widg.wy  $w.widg.wz -side left
	
	frame $w.fv -relief flat -borderwidth 2
	label $w.fv.head -text "Field value at the point:\t"
	label $w.fv.value
	pack $w.fv.head $w.fv.value -side left
	update_widg

	frame $w.switch -relief flat
	label $w.switch.head -text "Widget:"
	radiobutton $w.switch.none -text "None" -variable switch -value 0 -command "$this-c WidgetOff"
	radiobutton $w.switch.cross -text "Crosshair" -variable switch -value 1 -command "$this-c WidgetOn"
	
	$w.switch.cross select
	$this-c WidgetOn
	
	pack $w.switch.head $w.switch.cross $w.switch.none -side top -anchor w
	
        pack $w.text $w.switch $w.widg $w.fv  -side top -pady 5 -padx 7 -anchor w
    }
    
    method update_widg {} {
	if { $IsInit!=0} {
	set w .ui[modname]
	$w.widg.wx configure -text [set  $this-w_x]
	$w.widg.wy configure -text [set  $this-w_y]
	$w.widg.wz configure -text [set  $this-w_z]
	$w.fv.value configure -text [set $this-field_value]
	}
    }
    
    method update_field {} {
	if { $IsInit!=0 } {
	set w .ui[modname]
	wm resizable $w 1 1
	$w.text configure -state normal

	$w.text delete 1.0 end

	set header  "\n Scalar Field Parameters:\n"                                                                                           
	set minMess "\n Minimum Value:\t"
	set maxMess "\n Maximum Value:\t"
	set bbMess1  "\n Bounding Points:\t"
	set bbMess2  "\n \t\t"
	set gtMess  "\n Grid Type:\t"
	set nelemsMsg "\n # of elements:\t"

	append minMess [set $this-minValue]
	append maxMess [set $this-maxValue]	
	append bbMess1 [set $this-bnd1]
	append bbMess2 [set $this-bnd2]
	append gtMess  [set $this-gridType]
	append nelemsMsg  [set $this-num_elements]

	set cmp [list [string length $bbMess1] [string length $bbMess2] [string length $minMess] [string length $maxMess] [string length $gtMess] [string length $nelemsMsg]]
	set lth 45
	set i 0

	while {$i<6} {
	    if {[lindex $cmp $i] > $lth} {
		set lth [lindex $cmp $i]
	    }
	    incr i
	}

	$w.text configure -width $lth

	$w.text insert end $header
	$w.text insert end $minMess
	$w.text insert end $maxMess
	$w.text insert end $bbMess1
	$w.text insert end $bbMess2
	$w.text insert end $gtMess
	$w.text insert end $nelemsMsg
	
	$w.text configure -state disabled
	wm resizable $w 0 0
	}
    }

}






