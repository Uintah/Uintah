itcl_class Insight_DataIO_ImageFileReader {
    inherit Module
    constructor {config} {
        set name ImageFileReader
        set_defaults
    }

    method set_defaults {} {
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }
        toplevel $w
	makeOpenFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "$this-c needexecute; destroy $w" \
		-cancel "destroy $w" \
		-title "Open Image File" \

    }
}


