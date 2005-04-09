
catch {rename PointsReader ""}

itcl_class PointsReader {
    inherit Module
    constructor {config} {
	set name PointsReader
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
	makeFilebox $w $this-ptsname "$this-c needexecute" "destroy $w"
    }
}
