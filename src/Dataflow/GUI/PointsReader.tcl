
catch {rename PointsReader ""}

itcl_class PSECommon_Readers_PointsReader {
    inherit Module
    constructor {config} {
	set name PointsReader
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
	makeFilebox $w $this-ptsname "$this-c needexecute" "destroy $w"
    }
}
