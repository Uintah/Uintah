
catch {rename ColorMapReader ""}

itcl_class PSECommon_Readers_ColorMapReader {
    inherit Module
    constructor {config} {
	set name ColorMapReader
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
