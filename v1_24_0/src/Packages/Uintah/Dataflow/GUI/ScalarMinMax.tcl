
itcl_class Uintah_Operators_ScalarMinMax {
    inherit Module

    constructor {config} {
	set name ScalarMinMax
	set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	global $this-min_data
	global $this-max_data
	global $this-min_index
	global $this-max_index
	global $this-min_values
	global $this-max_values

	set $this-min_data "---"
	set $this-max_data "---"
	set $this-min_index "---"
	set $this-max_index "---"
	set $this-min_values "---"
	set $this-max_values "___"
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

	labelpair $lf.l1 "Min Value" $this-min_data
	labelpair $lf.l2 "Min index" $this-min_index
	labelpair $lf.l3 "\# Min Values" $this-min_values
	label $lf.l4 -text ""
	labelpair $lf.l5 "Max Value" $this-max_data
	labelpair $lf.l6 "Max index" $this-max_index
	labelpair $lf.l7 "\# Max Values" $this-max_values
	pack $lf.l1 $lf.l2 $lf.l3 $lf.l4 $lf.l5 \
	     $lf.l7 -side top -expand y -fill x

	makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 40 -anchor w -just left \
		-fore darkred -borderwidth 0
	pack $win.l1 $win.colon $win.l2 -side left
    } 
}

