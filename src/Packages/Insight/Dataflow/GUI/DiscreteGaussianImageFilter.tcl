itcl_class Insight_Filters_DiscreteGaussianImageFilter {
    inherit Module
    constructor {config} {
        set name DiscreteGaussianImageFilter

	global $this-variance
	global $this-max_error

        set_defaults
    }

    method set_defaults {} {
	set $this-variance 10
	set $this-max_error 0.001
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
	frame $w.a
	frame $w.a.labs
	frame $w.a.ents

	label $w.a.labs.variance -text "Variance" -just left
	entry $w.a.ents.variance -textvariable $this-variance
	label $w.a.labs.max_error -text "Max Error" -just left
	entry $w.a.ents.max_error -textvariable $this-max_error

	pack $w.a.labs.variance $w.a.labs.max_error -side top -anchor w
	pack $w.a.ents.variance $w.a.ents.max_error -side top -anchor w
	pack $w.a.labs $w.a.ents -side left

	frame $w.b
	button $w.b.execute -text "Execute" -command "$this-c needexecute"
	pack $w.b.execute -side top -e n -f both
	pack $w.b -side bottom

	pack $w.a -side left
	
    }
}
