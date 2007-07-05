catch {rename DaveW_Readers_TensorFieldReader ""}

itcl_class DaveW_Readers_TensorFieldReader {
    inherit Module
    constructor {config} {
	set name TensorFieldReader
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
