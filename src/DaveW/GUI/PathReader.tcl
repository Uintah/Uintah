catch {rename DaveW_Readers_PathReader ""}

itcl_class DaveW_Readers_PathReader {
    inherit Module
    constructor {config} {
	set name PathReader
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
