itcl_class SCIRun_Render_SynchronizeGeometry {
    inherit Module
   
    constructor {config} {
        set name SynchronizeGeometry
        set_defaults
    }
    method set_defaults {} {
	global $this-enforce
	
	set $this-enforce 1
	
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
	wm title $w "Geometry Barrier"
	
	checkbutton $w.enforce -text "Enforce Barrier " -variable $this-enforce \
	    -onvalue 1 -offvalue 0 -padx 5
	pack $w.enforce
	

    }
}


