# example of Writer
# by Samsonov Alexei
# October 2000 
# NOTE: if file to be splitted, uncomment corresponding lines in the file

catch {rename SCIRun_DataIO_ColorMapWriter ""}

itcl_class SCIRun_DataIO_ColorMapWriter {
    inherit Module
    constructor {config} {
	set name ColorMapWriter
	set_defaults
    }
    method set_defaults {} {
	global $this-filetype
	set $this-filetype Binary
	# set $this-split 0
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
	set defext ".cmap"
	
	# name to appear initially
	set defname "MyColorMap"
	set title "Save colormap file"

	# file types to appers in filter box
	set types {
	    {{Colormap}        {.cmap} }
	    {{All Files}       {.*}    }
	}
	
	######################################################
	
	makeSaveFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "$this-c needexecute; destroy $w" \
		-cancel "destroy $w" \
		-title $title \
		-filetypes $types \
	        -initialfile $defname \
		-initialdir $initdir \
		-defaultextension $defext \
		-formatvar $this-filetype 
		#-splitvar $this-split
    }
}
