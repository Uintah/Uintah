itcl_class Uintah_Operators_TensorToTensorConvertor {
    inherit Module
    constructor {config} {
        set name TensorToTensorConvertor
        set_defaults
    }

    method set_defaults {} {
        global $this-operation
        set $this-operation 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

        button $w.button -text Close -command "destroy $w"
        pack $w.button -side bottom -expand yes -fill x -padx 2 -pady 2

        frame $w.frame -relief raised -bd 1
        label $w.frame.label -text "Convert Into .."

        radiobutton $w.frame.none -text "None" \
                -variable $this-operation -value 0 

        radiobutton $w.frame.green -text "Green-Lagrange Strain Tensor" \
                -variable $this-operation -value 1 

        radiobutton $w.frame.cauchy -text "Cauchy-Green Deformation Tensor" \
                -variable $this-operation -value 2 

        radiobutton $w.frame.finger -text "Finger Deformation Tensor" \
                -variable $this-operation -value 3 

        pack $w.frame.label $w.frame.none $w.frame.green $w.frame.cauchy \
             $w.frame.finger

        pack $w.frame -side left -padx 2 -pady 2 -fill y

    }
}


