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



itcl_class CardioWave_Tools_CreateString {
	inherit Module

	constructor {config} {
		set name CreateString
		set_defaults
	}

	method set_defaults {} {
		global $this-inputstring
        global $this-labelstring
        set $this-inputstring ""
        set $this-labelstring "LABEL"
	}

	method ui {} {

		global $this-inputstring
        global $this-labelstring
		set w .ui[modname]

		# test whether gui is already out there
		# raise will move the window to the front
		# so the user can modify the settings

		if {[winfo exists $w]} {
			raise $w
			return
		}

		# create a new gui window

		toplevel $w 

        frame $w.frame
        pack $w.frame -side top -fill x -expand yes
        entry $w.frame.label -textvariable $this-labelstring
        entry $w.frame.string -textvariable $this-inputstring
        pack $w.frame.label -side left 
        pack $w.frame.string -side right -fill x -expand yes

        makeSciButtonPanel $w $w $this
	}

}
