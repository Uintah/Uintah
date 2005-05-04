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



itcl_class CardioWave_Tools_PrintF {
	inherit Module

	constructor {config} {
		set name PrintF
		set_defaults
	}

	method set_defaults {} {
		global $this-formatstring
		global $this-labelstring

        set $this-formatstring ""
        set $this-labelstring ""
	}

	method ui {} {

		global $this-formatstring
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
        frame $w.frame2
        pack $w.frame2 -side top -fill x -expand yes
        
        label $w.frame.label -text "Format string"
        entry $w.frame.string -textvariable $this-formatstring
        pack $w.frame.label -side left 
        pack $w.frame.string -side right -fill x -expand yes

        label $w.frame2.label -text "object name"
        entry $w.frame2.string -textvariable $this-labelstring
        pack $w.frame2.label -side left 
        pack $w.frame2.string -side right -fill x -expand yes

        makeSciButtonPanel $w $w $this
	}

}
