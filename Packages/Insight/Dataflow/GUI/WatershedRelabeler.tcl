 itcl_class Insight_Filters_WatershedRelabeler {
    inherit Module
    constructor {config} {
         set name WatershedRelabeler

         global $this-flood_level

         set_defaults
    }

    method set_defaults {} {

         set $this-flood_level 0.25    
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

        frame $w.flood_level
        label $w.flood_level.label -text "flood_level"
        entry $w.flood_level.entry -textvariable $this-flood_level
        pack $w.flood_level.label $w.flood_level.entry -side left
        pack $w.flood_level
        
        button $w.execute -text "Execute" -command "$this-c needexecute"
        button $w.close -text "Close" -command "destroy $w"
        pack $w.execute $w.close -side top

    }
}

