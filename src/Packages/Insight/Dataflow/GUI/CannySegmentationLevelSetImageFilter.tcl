 itcl_class Insight_Filters_CannySegmentationLevelSetImageFilter {
    inherit Module
    constructor {config} {
         set name CannySegmentationLevelSetImageFilter

         global $this-iterations
         global $this-negative_features
         global $this-max_rms_change
         global $this-threshold
         global $this-variance
         global $this-propagation_scaling
         global $this-advection_scaling
         global $this-curvature_scaling
         global $this-isovalue

         set_defaults
    }

    method set_defaults {} {

         set $this-iterations 10
         set $this-negative_features 1
         set $this-max_rms_change 0.001
         set $this-threshold 0.001
         set $this-variance 10
         set $this-propagation_scaling 1.0
         set $this-advection_scaling 1.0
         set $this-curvature_scaling 1.0
         set $this-isovalue 0.5    
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

        frame $w.iterations
        label $w.iterations.label -text "iterations"
        entry $w.iterations.entry -textvariable $this-iterations
        pack $w.iterations.label $w.iterations entry -side left
        pack $w.iterations

        frame $w.negative_features
        label $w.negative_features.label -text "negative_features"
        entry $w.negative_features.entry -textvariable $this-negative_features
        pack $w.negative_features.label $w.negative_features entry -side left
        pack $w.negative_features

        frame $w.max_rms_change
        label $w.max_rms_change.label -text "max_rms_change"
        entry $w.max_rms_change.entry -textvariable $this-max_rms_change
        pack $w.max_rms_change.label $w.max_rms_change entry -side left
        pack $w.max_rms_change

        frame $w.threshold
        label $w.threshold.label -text "threshold"
        entry $w.threshold.entry -textvariable $this-threshold
        pack $w.threshold.label $w.threshold entry -side left
        pack $w.threshold

        frame $w.variance
        label $w.variance.label -text "variance"
        entry $w.variance.entry -textvariable $this-variance
        pack $w.variance.label $w.variance entry -side left
        pack $w.variance

        frame $w.propagation_scaling
        label $w.propagation_scaling.label -text "propagation_scaling"
        entry $w.propagation_scaling.entry -textvariable $this-propagation_scaling
        pack $w.propagation_scaling.label $w.propagation_scaling entry -side left
        pack $w.propagation_scaling

        frame $w.advection_scaling
        label $w.advection_scaling.label -text "advection_scaling"
        entry $w.advection_scaling.entry -textvariable $this-advection_scaling
        pack $w.advection_scaling.label $w.advection_scaling entry -side left
        pack $w.advection_scaling

        frame $w.curvature_scaling
        label $w.curvature_scaling.label -text "curvature_scaling"
        entry $w.curvature_scaling.entry -textvariable $this-curvature_scaling
        pack $w.curvature_scaling.label $w.curvature_scaling entry -side left
        pack $w.curvature_scaling

        frame $w.isovalue
        label $w.isovalue.label -text "isovalue"
        entry $w.isovalue.entry -textvariable $this-isovalue
        pack $w.isovalue.label $w.isovalue entry -side left
        pack $w.isovalue
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

