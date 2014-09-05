itcl_class SCIRun_Visualization_AddLight {
    inherit Module
    constructor {config} {
        set name AddLight
        set_defaults
    }

    method set_defaults {} {
	global $this-control_pos_saved
	global $this-control_x
	global $this-control_y
	global $this-control_z
	global $this-at_x
	global $this-at_y
	global $this-at_z
	global $this-type
	global $this-on
	set $this-type 0
	set $this-on 1
	set $this-control_pos_saved 0
	set $this-control_x 0
	set $this-control_y 0
	set $this-control_z 0
	set $this-at_x 0
	set $this-at_y 0
	set $this-at_z 1
	
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	
	set n "$this-c needexecute"

	make_labeled_radio $w.light_type "Light type:" $n left \
	    $this-type { {Directional 0} {Point 1} {Spot 2}}
	checkbutton  $w.on  -text "on/off" \
	    -variable $this-on \
	    -command $n
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.light_type $w.on $w.close -side top
    }
}


