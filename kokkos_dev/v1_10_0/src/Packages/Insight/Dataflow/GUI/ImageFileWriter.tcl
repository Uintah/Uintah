itcl_class Insight_DataIO_ImageFileWriter {
    inherit Module
    constructor {config} {
        set name ImageFileWriter
        set_defaults
    }

    method set_defaults {} {
	global $this-filetype
	set $this-filetype Binary
	set $this-split 0
    }

    method ui {} {
        set w .ui[modname]
	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }

	############
	set types {
	    {{Meta Image}        {.mhd} }
	    {{PNG Image}        {.png} }
	    {{All Files}       {.*} }
	}
	set defname "MyImage.mhd"
	set defext ".mhd"
	############
        toplevel $w
	makeSaveFilebox \
	    -parent $w \
	    -filevar $this-FileName \
	    -command "$this-c needexecute; destroy $w" \
	    -cancel "destroy $w" \
	    -title "Save Image File" \
	    -filetypes $types \
	    -initialfile $defname \
	    -defaultextension $defext \
	    -formatvar $this-filetype \
	    -splitvar $this-split
    }
}


