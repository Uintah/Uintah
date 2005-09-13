#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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


# GUI for NrrdReader module
# by Samsonov Alexei
# December 2000

catch {rename Teem_DataIO_NrrdReader ""}

itcl_class Teem_DataIO_NrrdReader {
    inherit Module
    constructor {config} {
	set name NrrdReader
	set_defaults
    }

    method set_defaults {} {
	global $this-filename
        global $this-types
        global $this-filetype

	set $this-filename ""
    }

    method ui {} {
	global env
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
	
	# extansion to append if no extension supplied by user
	set defext ".nrrd"
	set title "Open nrrd file"

	######################################################
	
        # Unwrap $this-types into a list.
	set tmp1 [set $this-types]
	set tmp2 [eval "set tmp3 $tmp1"]

	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -setcmd "wm withdraw $w" \
	    -command "$this-c needexecute; wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title $title \
	    -filetypes $tmp2 \
	    -initialdir $initdir \
	    -defaultextension $defext \
            -selectedfiletype $this-filetype

	moveToCursor $w
    }
}
