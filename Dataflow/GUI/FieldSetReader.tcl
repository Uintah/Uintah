# GUI for FieldSetReader module
# by Samsonov Alexei
# December 2000

catch {rename SCIRun_DataIO_FieldSetReader ""}

itcl_class SCIRun_DataIO_FieldSetReader {
    inherit Module
    constructor {config} {
	set name FieldSetReader
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
	if {[info exists env(PSE_DATA)]} {
	    set initdir $env(PSE_DATA)
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
	set defext ".fset"
	set title "Open field file"
	
	# file types to appers in filter box
	set types {
	    {{FieldSet File}     {.fset}      }
	    {{All Files} {.*}   }
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
