
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
        global $this-extraCells_x
        global $this-extraCells_y
        global $this-extraCells_z
        
        set $this-field_name "---"
        set $this-field_time "---"
        set $this-output_choice 2
        set $this-extraCells_x 0
        set $this-extraCells_y 0
        set $this-extraCells_z 0
    }

    method ui {} {
	
	set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	wm minsize $w 200 100
        
        #  Variable name and time
        frame $w.f1
        pack  $w.f1
        label_pair $w.f1.field_name "Input Field Name" $this-field_name 20
        label_pair $w.f1.field_time "Input Field Time" $this-field_time 20

        # include extra cells?
        frame $w.f2 -borderwidth 2 -relief groove
        pack  $w.f2
        set the_com "$this-c needexecute"
        
        label $w.f2.header -text "Include extra cells in difference calculation"
        checkbutton $w.f2.extraCells_x  -text "(x)" -variable $this-extraCells_x -command $the_com
        checkbutton $w.f2.extraCells_y  -text "(y)" -variable $this-extraCells_y -command $the_com
        checkbutton $w.f2.extraCells_z  -text "(z)" -variable $this-extraCells_z -command $the_com
        
        # Select the type of output you'd like
        frame $w.output_choice -borderwidth 2 -relief groove
        set the_var "$this-output_choice"
        radiobutton $w.output_choice.original  -text "1) Original Data"                    -value 0  -variable $the_var -command $the_com 
        radiobutton $w.output_choice.exact     -text "2) Method of Manufactured Solution"  -value 1  -variable $the_var -command $the_com
        radiobutton $w.output_choice.diff      -text "Difference (1) - (2)"                -value 2  -variable $the_var -command $the_com
        
        pack $w.output_choice.original $w.output_choice.exact $w.output_choice.diff -anchor w
        pack $w.f2.header -padx 10
        pack $w.f2.extraCells_x $w.f2.extraCells_y $w.f2.extraCells_z  -anchor w

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

