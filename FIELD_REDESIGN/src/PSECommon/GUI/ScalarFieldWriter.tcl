
catch {rename ScalarFieldWriter ""}

itcl_class PSECommon_Writers_ScalarFieldWriter {
    inherit Module
    constructor {config} {
	set name ScalarFieldWriter
	set_defaults
    }
    method set_defaults {} {
	global $this-filetype
	global $this-split

	set $this-filetype Binary
	set $this-split 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w

	make_labeled_radio $w.filetype "Format:" "" left $this-filetype \
		{Binary ASCII}
	checkbutton $w.split -text "Split" -relief flat \
		-variable $this-split

	pack $w.filetype $w.split
	entry $w.f -textvariable $this-filename -width 40 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom
	bind $w.f <Return> "$this-c needexecute "
    }
}
