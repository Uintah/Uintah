# example of reader
# by Samsonov Alexei
# October 2000


catch {rename SCIRun_DataIO_ColorMapReader ""}

itcl_class SCIRun_DataIO_ColorMapReader {
    inherit Module
    constructor {config} {
	set name ColorMapReader
	set_defaults
    }

    method set_defaults {} {
	global $this-filetype
    }

    method ui {} {
	global env
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]

	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
	}

	toplevel $w
	set initdir ""
	
	# place to put preferred data directory
	# it's used if $this-filename is empty
	if {[info exists env(SCIRUN_DATA)]} {
	    set initdir $env(SCIRUN_DATA)
	}
	
	if { $initdir==""} {
	    if {[info exists env(SCI_DATA)]} {
		set initdir $env(SCI_DATA)
	    } elseif {[info exists env(SCIRUN_DATA)]} {
		set initdir $env(SCIRUN_DATA)
	    }
	}
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".cmap"
	set title "Open colormap file"
	
	# file types to appers in filter box
	set types {
	    {{Color Map}        {.cmap} }
	    {{All Files}       {.*}    }
	}
	
	######################################################
	
	makeOpenFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "$this-c needexecute; destroy $w" \
		-cancel "destroy $w" \
		-title $title \
		-filetypes $types \
		-initialdir $initdir \
		-defaultextension $defext
    }
}
