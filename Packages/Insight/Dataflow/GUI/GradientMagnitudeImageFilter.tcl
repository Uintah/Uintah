 itcl_class Insight_Filters_GradientMagnitudeImageFilter {
    inherit Module
    constructor {config} {
         set name GradientMagnitudeImageFilter


         set_defaults
    }

    method set_defaults {} {
    
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
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

