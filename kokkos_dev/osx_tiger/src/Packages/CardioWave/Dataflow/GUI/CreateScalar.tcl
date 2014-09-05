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



itcl_class CardioWave_Tools_CreateScalar {
	inherit Module

	constructor {config} {
		set name CreateScalar
		set_defaults
	}

	method set_defaults {} {
		global $this-inputscalar
        global $this-labelstring
        global $this-typestring
        set $this-inputscalar "0.0"
        set $this-labelstring "LABEL"
        set $this-scalartype "double"
	}

	method ui {} {

		global $this-inputscalar
        global $this-labelstring
        global $this-scalartype
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
        pack $w.frame2 -side top -anchor e
        
        entry $w.frame.label -textvariable $this-labelstring
        entry $w.frame.string -textvariable $this-inputscalar
        pack $w.frame.label -side left 
        pack $w.frame.string -side right -fill x -expand yes

        iwidgets::optionmenu $w.frame2.type -command "$this Synchronise"
			foreach dformat { {double} {float} {int8} {uint8} {int16} {uint16} {int32} {uint32}} {
				$w.frame2.type insert end $dformat
			}
            
        set dataformatindex [lsearch {{double} {float} {int8} {uint8} {int16} {uint16} {int32} {uint32}} [set $this-scalartype] ]
        if [expr $dataformatindex >= 0] { $w.frame2.type select $dataformatindex }

        pack $w.frame2.type -side top -anchor e
        makeSciButtonPanel $w $w $this
	}


    method Synchronise {} {

        global $this-scalartype
		set w .ui[modname]
        
		if {[winfo exists $w]} {
            set $this-scalartype [$w.frame2.type get]
        }
        
    }
}
