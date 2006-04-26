itcl_class ModelCreation_TensorVectorMath_DataArrayMeasure {
    inherit Module
    constructor {config} {
        set name DataArrayMeasure
        set_defaults
    }

    method set_defaults {} {
        global $this-measure
        set $this-measure "Sum"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 170 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        make_labeled_radio $w.f.r "DataArray measures:" "" \
                top $this-measure \
                {{"Sum" Sum}\
                {"Mean" Mean}\
                {"Variance" Variance}\
                {"Std Deviation" StdDev}\
                {"Norm" Norm}\
                {"Maximum" Maximum}\
                {"Minimum" Minimum}\
                {"Median" Median}}
        pack $w.f.r -side top -expand 1 -fill x
        pack $w.f -expand 1 -fill x
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
     }
}


