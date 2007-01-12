
itcl_class Uintah_Operators_CompareMMS {
    inherit Module

    constructor {config} {
	set name CompareMMS
	set_defaults
    }

    method set_defaults {} {
        global $this-field_name
        global $this-field_time
        global $this-output_choice
        
        set $this-field_name "---"
        set $this-field_time "---"
        set $this-output_choice 2
    }

    method ui {} {
	
	set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	wm minsize $w 400 150

        frame $w.f1
        pack  $w.f1

        label_pair $w.f1.field_name "Input Field Name" $this-field_name 20
        label_pair $w.f1.field_time "Input Field Time" $this-field_time 20

        frame $w.output_choice -borderwidth 2 -relief groove

        set the_var "$this-output_choice"
        set the_com "$this-c needexecute"
        radiobutton $w.output_choice.original  -text "Original"    -value 0  -variable $the_var -command $the_com 
        radiobutton $w.output_choice.exact     -text "Exact"       -value 1  -variable $the_var -command $the_com
        radiobutton $w.output_choice.diff      -text "Difference"  -value 2  -variable $the_var -command $the_com
        pack $w.output_choice.original $w.output_choice.exact $w.output_choice.diff -anchor w

        pack $w.f1.field_name $w.f1.field_time -padx 10
        pack $w.output_choice -padx 10

        # add frame for SCI Button Panel
        frame $w.control -relief flat
        pack $w.control -side top -expand yes -fill both
	makeSciButtonPanel $w.control $w $this
	moveToCursor $w
    }

    method set_to_exact {} {
        set $this-output_choice 1
        # The following line allows the button to change before the blinking takes place.
        update idletasks
	set w .ui[modname]
        $w.output_choice.exact flash
        $w.output_choice.exact flash
    }

}
# end class CompareMMS

