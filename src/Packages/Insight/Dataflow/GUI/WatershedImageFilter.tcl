 itcl_class Insight_Filters_WatershedImageFilter {
    inherit Module
    constructor {config} {
         set name WatershedImageFilter

         global $this-threshold
         global $this-level

         set_defaults
    }

    method set_defaults {} {

         set $this-threshold 0.01
         set $this-level 0.1    
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

        frame $w.threshold
        label $w.threshold.label -text "threshold"
        entry $w.threshold.entry -textvariable $this-threshold
        pack $w.threshold.label $w.threshold.entry -side left
        pack $w.threshold

        frame $w.level
        label $w.level.label -text "level"
        entry $w.level.entry -textvariable $this-level
        pack $w.level.label $w.level.entry -side left
        pack $w.level
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

