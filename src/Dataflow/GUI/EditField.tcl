itcl_class SCIRun_Fields_EditField {
    inherit Module
    constructor {config} {
        set name EditField
        set_defaults
    }

    method set_defaults {} {
	global $this-name
	global $this-bboxmin
	global $this-bboxmax
	global $this-typename
	global $this-datamin
	global $this-datamax
	global $this-numnodes
	global $this-numelems
	global $this-dataat
	set $this-name "---"
	set $this-bboxmin "---"
	set $this-bboxmax "---"
	set $this-typename "---"
	set $this-datamin "---"
	set $this-datamax "---"
	set $this-numnodes "---"
	set $this-numelems "---"
	set $this-dataat "---"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	wm minsize $w 416 408
	wm maxsize $w 416 1000

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Attributes" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
	labelpair $att.l1 "Name" "[set $this-name]"
	labelpair $att.l2 "Typename" "[set $this-typename]"
	labelpair $att.l3 "BBox min" "[set $this-bboxmin]"
	labelpair $att.l4 "BBox max" "[set $this-bboxmax]"
	labelpair $att.l5 "Data min" "[set $this-datamin]"
	labelpair $att.l6 "Data max" "[set $this-datamax]"
	labelpair $att.l7 "# Nodes" "[set $this-numnodes]"
	labelpair $att.l8 "# Elements" "[set $this-numelems]"
	labelpair $att.l9 "Data at" "[set $this-dataat]"
	pack $att.l1 $att.l2 $att.l3 $att.l4 $att.l5 \
	     $att.l6 $att.l7 $att.l8 $att.l9 -side top 

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Attributes" 
			       
	pack $w.edit 
	set edit [$w.edit childsite]
	
	labelentry $edit.l1 "Name" "[set $this-name]"
	labelcombo $edit.l2 "Typename" "[set $this-typename]"
	labelentry $edit.l3 "BBox min" "[set $this-bboxmin]"
	labelentry $edit.l4 "BBox max" "[set $this-bboxmax]"
	labelentry $edit.l5 "Data min" "[set $this-datamin]"
	labelentry $edit.l6 "Data max" "[set $this-datamax]"
	labelentry $edit.l7 "# Nodes" "[set $this-numnodes]"
	labelentry $edit.l8 "# Elements" "[set $this-numelems]"
	labelcombo $edit.l9 "Data at" {Field::CELL Field::FACE Field::EDGE \
		                       Field::NODE Field::NONE}
	pack $edit.l1 $edit.l2 $edit.l3 $edit.l4 $edit.l5 \
	     $edit.l6 $edit.l7 $edit.l8 $edit.l9 -side top 
    }

    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width 10 -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -text $text2 -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelentry { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width 10 -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 40 -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    }

    method labelcombo { win text1 arglist } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width 10 -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left
	iwidgets::optionmenu $win.c
	set i 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
	    $win.c insert end $elem
	}
	label $win.l2 -text "" -width 40 -anchor w -just left
	pack $win.l1 $win.colon $win.c $win.l2 -side left	
    }

    method update_attributes {} {
	set w .ui[modname]
	set att [$w.att childsite]
	config_labelpair $att.l1 "[set $this-name]"
	config_labelpair $att.l2 "[set $this-typename]"
	config_labelpair $att.l3 "[set $this-bboxmin]"
	config_labelpair $att.l4 "[set $this-bboxmax]"
	config_labelpair $att.l5 "[set $this-datamin]"
	config_labelpair $att.l6 "[set $this-datamax]"
	config_labelpair $att.l7 "[set $this-numnodes]"
	config_labelpair $att.l8 "[set $this-numelems]"
	config_labelpair $att.l9 "[set $this-dataat]"
    }

    method config_labelpair { win text2 } {
	$win.l2 configure -text $text2
    }
}


