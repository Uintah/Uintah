
catch {rename Nrrd_DataIO_NrrdReader ""}

itcl_class Nrrd_DataIO_NrrdReader {
    inherit Module
    constructor {config} {
	set name NrrdReader
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
