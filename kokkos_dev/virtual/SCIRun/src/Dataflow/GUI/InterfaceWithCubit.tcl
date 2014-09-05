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


itcl_class SCIRun_NewField_InterfaceWithCubit {
    inherit Module

    constructor {config} {
        set name InterfaceWithCubit
    }

    method choose_ncdump {} {
        set w .ui[modname]-choose
        if {[winfo exists $w]} { 
	    SciRaise $w
	    return 
        }

        # file types to appers in filter box
        set types { 
	    { {All Files} {*} }
	}
        makeOpenFilebox \
	    -parent [toplevel $w -class TkFDialog] \
	    -filevar $this-ncdump \
	    -command "wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title "Choose NetCDF ncdump executable" \
	    -filetypes $types \
	    -initialdir /usr/local/bin 
        moveToCursor $w
	SciRaise $w
    }



    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.d
	pack $w.d -side top -e y -f both -padx 5 -pady 5	
	label $w.d.l -text "Directoy containing claro: "
	entry $w.d.e -textvariable $this-cubitdir
	pack $w.d.l $w.d.e -side left

	frame $w.e
	pack $w.e -side top -e y -f both -padx 5 -pady 5	
	label $w.e.l -text "Path to ncdump: "
	entry $w.e.e -textvariable $this-ncdump
	button $w.e.b -text Browse... -command "$this choose_ncdump"
	pack $w.e.l $w.e.e $w.e.b -side left



	makeSciButtonPanel $w $w $this

	moveToCursor $w
    }
}


