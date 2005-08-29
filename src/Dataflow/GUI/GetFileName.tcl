itcl_class SCIRun_String_GetFileName {
    inherit Module
    constructor {config} {
        set name GetFileName
        set_defaults
    }

    method set_defaults {} {
    		global $this-filename
        set $this-filename ""
    }

    method ui {} {

    		global $this-filename
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w -class TkFDialog

        set initdir ""
      
        # place to put preferred data directory
        # it's used if $this-filename is empty
      
        # Use the standard data dirs
        # I guess there is no .mat files in there
        # at least not yet

        if {[info exists env(SCIRUN_DATA)]} {
              set initdir $env(SCIRUN_DATA)
        } elseif {[info exists env(SCI_DATA)]} {
              set initdir $env(SCI_DATA)
        } elseif {[info exists env(PSE_DATA)]} {
              set initdir $env(PSE_DATA)
        }
      
        makeOpenFilebox \
            -parent $w \
            -filevar $this-filename \
            -command "$this-c needexecute; wm withdraw $w" \
            -commandname "Execute" \
            -cancel "wm withdraw $w" \
            -title "Select file" \
            -filetypes {{ "All files" "*.*" } }\
            -initialdir $initdir \
            -defaultextension "*.*" \
            -allowMultipleFiles $this \
            -selectedfiletype 0

        moveToCursor $w
    }
}


