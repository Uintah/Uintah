catch {rename Packages/DaveW_Readers_ContourSetReader ""}

itcl_class Packages/DaveW_Readers_ContourSetReader {
    inherit Module
    constructor {config} {
	set name ContourSetReader
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
