
catch {rename ColumnMatrixReader ""}

itcl_class SCIRun_DataIO_ColumnMatrixReader {
    inherit Module
    constructor {config} {
	set name ColumnMatrixReader
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
