# Ptolemy tcl-based GUI

catch {rename Ptolomy_Vergil_VergilApplication ""}

itcl_class Ptolemy_Vergil_VergilApplication {
  inherit Module
  
  constructor { config } {
    set name Vergil
    set_defaults
  }

  method set_defaults {} {
    ##global $this-types
    global $this-filename
    ##global $this-filetype

    set $this-filename ""
    ##set $this-filetype ""
  }

  method ui {} {
    global $this-filename
    set w .ui[modname]

    if {[winfo exists $w]} {
      return
    }

    toplevel $w -class TkFDialog
    # place to put preferred data directory
    # it's used if $this-filename is empty
    set initdir [netedit getenv SCIRUN_DATA]

    #######################################################
    # to be modified for particular reader

    # extension to append if no extension supplied by user
    set defext ".xml"
    set title "Open moml (Ptolemy model) file"

    # file types to appers in filter box
    set types {
      {{Ptolemy Module}  {.xml} }
      {{All Files}       {.*}   }
    }

    ######################################################
    
    makeOpenFilebox \
        -parent $w \
        -filevar $this-filename \
        -setcmd "wm withdraw $w" \
        -command "$this-c needexecute; wm withdraw $w" \
        -cancel "wm withdraw $w" \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext

    moveToCursor $w 
  }
}

