 itcl_class Insight_Filters_DiscreteGaussianImageFilter {
    inherit Module
    constructor {config} {
         set name DiscreteGaussianImageFilter

         global $this-variance
         global $this-maximum_error

         set_defaults
    }

    method set_defaults {} {

         set $this-variance 10
         set $this-maximum_error 0.001    
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }
        toplevel $w

        frame $w.variance
        label $w.variance.label -text "variance"
        entry $w.variance.entry -textvariable $this-variance
        pack $w.variance.label $w.variance.entry -side left
        pack $w.variance

        frame $w.maximum_error
        label $w.maximum_error.label -text "maximum_error"
        entry $w.maximum_error.entry -textvariable $this-maximum_error
        pack $w.maximum_error.label $w.maximum_error.entry -side left
        pack $w.maximum_error
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

