#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# example of Writer
# by Samsonov Alexei
# October 2000 
# NOTE: if file to be splitted, uncomment corresponding lines in the file


itcl_class SCIRun_DataIO_WriteBundle {
    inherit Module

    constructor {config} {
        set name WriteBundle
    }

    method overwrite {} {
        global $this-confirm $this-confirm-once $this-filename $this-old-filename
				
				if {![info exists $this-old-filename]} {
					set $this-old-filename ""
				}
				
				if {[set $this-confirm-once] == 1} {
					if { [string equal [set $this-filename] [set $this-old-filename] ]} {
					  return 1
					}
				}
        if {[info exists $this-confirm] && [info exists $this-filename] && \
                [set $this-confirm] && [file exists [set $this-filename]] } {
            set value [tk_messageBox -type yesno -parent . \
                           -icon warning -message \
                           "File [set $this-filename] already exists.\n Would you like to overwrite it?"]
            if [string equal "no" $value] { return 0 }
        }
				
				set $this-old-filename [set $this-filename]
        return 1
    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            # Refresh UI
            biopseFDialog_RefreshCmd $w
            return
        }
        
        toplevel $w -class TkFDialog
        # place to put preferred data directory
        # it's used if $this-filename is empty
        set initdir [netedit getenv SCIRUN_DATA]

        #######################################################
        # to be modified for particular reader

        # extansion to append if no extension supplied by user
        set defext ".bdl"
        
        # name to appear initially
        set defname "MyBundle"
        set title "Save bundle file"

        # Unwrap $this-types into a list.
        set tmp1 [set $this-types]
        set tmp2 [eval "set tmp3 $tmp1"]
        
        ######################################################
        
        makeSaveFilebox \
            -parent $w \
            -filevar $this-filename \
            -setcmd "wm withdraw $w" \
            -command "$this-c needexecute; wm withdraw $w" \
            -cancel "wm withdraw $w" \
            -title $title \
            -filetypes $tmp2 \
            -initialfile $defname \
            -initialdir $initdir \
            -defaultextension $defext \
            -formatvar $this-filetype \
            -confirmvar $this-confirm \
            -confirmoncevar $this-confirm-once \
            -selectedfiletype $this-exporttype

        moveToCursor $w
    }
}
