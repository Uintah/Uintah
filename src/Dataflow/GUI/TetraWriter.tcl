
catch {rename TetraWriter ""}

itcl_class PSECommon_Writers_TetraWriter {
    inherit Module
    constructor {config} {
	set name TetraWriter
	set_defaults
    }
    method set_defaults {} {
	global $this-filetype
	set $this-filetype Binary
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w

	label $w.label -text "Enter prefix for point and tetra files"
	pack $w.label
	entry $w.f -textvariable $this-filename -width 40 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom
	bind $w.f <Return> "$this-c needexecute "
    }
}
