itcl_class Insight_DataIO_ImageReaderFloat2D {
    inherit Module
    constructor {config} {
        set name ImageReaderFloat2D
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

	set defext ".mhd"
	set title "Open image file"
	
	# file types to appers in filter box
	set types {
	    {{Meta Image}        {.mhd} }
	    {{PNG Image}        {.png} }
	    {{All Files}       {.*}    }
	}

	makeOpenFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "$this-c needexecute; destroy $w" \
		-cancel "destroy $w" \
		-title "Open Image File" \
                -filetypes $types \
		-defaultextension $defext

    }
}


