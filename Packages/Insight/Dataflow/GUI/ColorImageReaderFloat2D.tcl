itcl_class Insight_DataIO_ColorImageReaderFloat2D {
    inherit Module
    constructor {config} {
        set name ColorImageReaderFloat2D
        set_defaults
    }

    method set_defaults {} {
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    return
        }
        toplevel $w -class TkFDialog

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
		-filevar $this-FileName \
		-command "$this-c needexecute; wm withdraw $w" \
		-cancel "wm withdraw $w" \
		-title "Open Image File" \
                -filetypes $types \
		-defaultextension $defext
    }
}


