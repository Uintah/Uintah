itcl_class Insight_Filters_DiscreteGaussianImageFilter {
    inherit Module
    constructor {config} {
        set name DiscreteGaussianImageFilter

	global $this-Variance
	global $this-MaximumError

        set_defaults
    }

    method set_defaults {} {
	set $this-Variance 10
	set $this-MaximumError 0.001
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
                                                   
	frame $w.Variance
	label $w.Variance.label -text "Variance" 
	entry $w.Variance.entry -textvariable $this-Variance
	pack $w.Variance.label $w.Variance.entry -side left 
	pack $w.Variance 

	frame $w.MaximumError
	label $w.MaximumError.label -text "MaximumError" 
	entry $w.MaximumError.entry -textvariable $this-MaximumError
	pack $w.MaximumError.label $w.MaximumError.entry -side left
	pack $w.MaximumError 


	button $w.execute -text "Execute" -command "$this-c needexecute"
	button $w.close -text "Close" -command "destroy $w"
	pack $w.execute $w.close 


    }
}
