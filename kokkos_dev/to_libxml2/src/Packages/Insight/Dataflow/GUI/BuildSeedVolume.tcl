#
#   For more information, please see: http://software.sci.utah.edu
#
#   The MIT License
#
#   Copyright (c) 2004 Scientific Computing and Imaging Institute,
#   University of Utah.
#
#   License for the specific language governing rights and limitations under
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#



 itcl_class Insight_Converters_BuildSeedVolume {
    inherit Module
    constructor {config} {
         set name BuildSeedVolume

         global $this-inside_value
         global $this-outside_value

         set_defaults
    }

    method set_defaults {} {
         set $this-inside_value 1
         set $this-outside_value 0
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    SciRaise $w
	    return
        }

        toplevel $w

        frame $w.inside_value
        label $w.inside_value.label -text "inside_value"
        entry $w.inside_value.entry \
            -textvariable $this-inside_value
        pack $w.inside_value.label $w.inside_value.entry -side left
        pack $w.inside_value

        frame $w.outside_value
        label $w.outside_value.label -text "outside_value"
        entry $w.outside_value.entry \
            -textvariable $this-outside_value
        pack $w.outside_value.label $w.outside_value.entry -side left
        pack $w.outside_value

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

}
