
itcl_class Uintah_Operators_ScalarMinMax {
    inherit Module

    constructor {config} {
	set name ScalarMinMax
	set_defaults
    }

    method set_defaults {} {

	global $this-field_name
	global $this-min_data
	global $this-max_data
	global $this-min_index
	global $this-max_index
	global $this-min_values
	global $this-max_values

	set $this-field_name "---"
	set $this-min_data "---"
	set $this-max_data "---"
	set $this-min_index "---"
	set $this-max_index "---"
	set $this-min_values "---"
	set $this-max_values "---"
    }

    method ui {} {
	
	set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	iwidgets::Labeledframe $w.lf -labelpos nw \
		               -labeltext "Min Max Info" 
	pack $w.lf
	set lf [$w.lf childsite]

	label_pair $lf.l0 "Field Name" $this-field_name
	frame $lf.separator0 -height 2 -relief sunken -borderwidth 2
        
	label_pair $lf.l1 "Min Value" $this-min_data
	label_pair $lf.l2 "Min index" $this-min_index
	label_pair $lf.l3 "\# Min Values" $this-min_values

	frame $lf.separator1 -height 2 -relief sunken -borderwidth 2

	label_pair $lf.l5 "Max Value" $this-max_data
	label_pair $lf.l6 "Max index" $this-max_index
	label_pair $lf.l7 "\# Max Values" $this-max_values

	pack $lf.l0 $lf.separator0 $lf.l1 $lf.l2 $lf.l3 $lf.separator1 $lf.l5 $lf.l6 $lf.l7 \
	       -side top -expand y -fill x -padx 5

      # add frame for SCI Button Panel
        frame $w.control -relief flat
        pack $w.control -side top -expand yes -fill both
	makeSciButtonPanel $w.control $w $this
	moveToCursor $w
    }

}

