 itcl_class Insight_Filters_GradientAnisotropicDiffusionImageFilter {
    inherit Module
    constructor {config} {
         set name GradientAnisotropicDiffusionImageFilter

         global $this-time_step
         global $this-iterations
         global $this-conductance_parameter

         set_defaults
    }

    method set_defaults {} {

         set $this-time_step 0.25
         set $this-iterations 10
         set $this-conductance_parameter 1    
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

        frame $w.time_step
        label $w.time_step.label -text "time_step"
        entry $w.time_step.entry -textvariable $this-time_step
        pack $w.time_step.label $w.time_step.entry -side left
        pack $w.time_step

        frame $w.iterations
        label $w.iterations.label -text "iterations"
        entry $w.iterations.entry -textvariable $this-iterations
        pack $w.iterations.label $w.iterations.entry -side left
        pack $w.iterations

        frame $w.conductance_parameter
        label $w.conductance_parameter.label -text "conductance_parameter"
        entry $w.conductance_parameter.entry -textvariable $this-conductance_parameter
        pack $w.conductance_parameter.label $w.conductance_parameter.entry -side left
        pack $w.conductance_parameter
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

