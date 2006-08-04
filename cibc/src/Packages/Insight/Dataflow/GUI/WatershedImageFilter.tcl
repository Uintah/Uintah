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

#############################################################
#  This is an automatically generated file for the 
#  itk::WatershedImageFilter
#############################################################


 itcl_class Insight_Filters_WatershedImageFilter {
    inherit Module
    constructor {config} {
         set name WatershedImageFilter

         global $this-threshold
         global $this-level

         set_defaults
    }

    method set_defaults {} {

         set $this-threshold 0.01
         set $this-level 0.1
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }

        toplevel $w


        frame $w.threshold
        label $w.threshold.label -text "threshold"
        entry $w.threshold.entry \
            -textvariable $this-threshold
        pack $w.threshold.label $w.threshold.entry -side left
        pack $w.threshold


        frame $w.level
        scale $w.level.scale -label  "level" \
           -variable $this-level \
           -from 0.0 -to 1.0 -orient horizontal  \
           -resolution 0.001
        pack $w.level.scale -side left
        pack $w.level
        
        frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top 

    }

}
