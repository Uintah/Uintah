
catch {rename ICPackages/NektarReader ""}

itcl_class Packages/Nektar_Readers_ICPackages/NektarReader {
    inherit Module
    constructor {config} {
	set name ICPackages/NektarReader
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
