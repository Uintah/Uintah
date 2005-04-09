
catch {rename ReadSonos ""}

itcl_class ReadSonos {
    inherit Module
    constructor {config} {
	set name ReadSonos
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	makeFilebox $w $this-filename "$this-c needexecute" "destroy $w"
    }
}
