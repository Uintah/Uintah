
catch {rename Downsample ""}

itcl_class PSECommon_Fields_Downsample {
    inherit Module
    constructor {config} {
	set name Downsample
	set_defaults
    }
    method set_defaults {} {
	global $this-downsamplex
	set $this-downsamplex 1
	global $this-downsampley
	set $this-downsampley 1
	global $this-downsamplez
	set $this-downsamplez 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w

	make_labeled_radio $w.downsamplex "X downsample: " "$this-c needexecute" left $this-downsamplex \
	    {1 2 4 8 16 32 64}
	make_labeled_radio $w.downsampley "Y downsample: " "$this-c needexecute" left $this-downsampley \
	    {1 2 4 8 16 32 64}
	make_labeled_radio $w.downsamplez "Z downsample: " "$this-c needexecute" left $this-downsamplez \
	    {1 2 4 8 16 32 64}
	pack $w.downsamplex $w.downsampley $w.downsamplez
    }
}

