
catch {rename VoidStarReader ""}

itcl_class VoidStarReader {
    inherit Module
    constructor {config} {
	set name VoidStarReader
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	makeFilebox $w $this-filename "$this-c needexecute" "destroy $w"
    }
}
