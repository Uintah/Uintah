catch {rename DaveW_FEM_RemapVector ""}

itcl_class DaveW_FEM_RemapVector {
    inherit Module

    constructor {config} {
	set name RemapVector
	set_defaults
    }


    method set_defaults {} {	
        global $this-zeroGround
	set $this-zeroGround 0
    }


    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
        toplevel $w
	wm minsize $w 100 50
	global $this-zeroGround
	checkbutton $w.b -text "Ground the zero'th value?" -variable $this-zeroGround
	pack $w.b
    }
	
}

