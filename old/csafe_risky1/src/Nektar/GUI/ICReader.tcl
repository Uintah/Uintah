
catch {rename ICNektarReader ""}

itcl_class Nektar_Readers_ICNektarReader {
    inherit Module
    constructor {config} {
	set name ICNektarReader
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
