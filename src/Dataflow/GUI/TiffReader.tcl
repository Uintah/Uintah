
catch {rename TiffReader ""}

itcl_class SCIRun_Image_TiffReader {
    inherit Module
    constructor {config} {
	set name TiffReader
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
