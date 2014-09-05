
catch {rename GeometryReader ""}

itcl_class PSECommon_Readers_GeometryReader {
    inherit Module
    constructor {config} {
	set name GeometryReader
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
