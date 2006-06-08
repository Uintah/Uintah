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
#  itk::BinaryThresholdImageFilter
#############################################################


 itcl_class Insight_Filters_BinaryThresholdImageFilter {
    inherit Module
    constructor {config} {
         set name BinaryThresholdImageFilter

         global $this-lower_threshold
         global $this-upper_threshold
         global $this-inside_value
         global $this-outside_value

         set_defaults
    }

    method set_defaults {} {

         set $this-lower_threshold 5.0
         set $this-upper_threshold 1.0
         set $this-inside_value 1
         set $this-outside_value 0
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


        frame $w.lower_threshold
        label $w.lower_threshold.label -text "lower_threshold"
        entry $w.lower_threshold.entry \
            -textvariable $this-lower_threshold
        pack $w.lower_threshold.label $w.lower_threshold.entry -side left
        pack $w.lower_threshold

        frame $w.upper_threshold
        label $w.upper_threshold.label -text "upper_threshold"
        entry $w.upper_threshold.entry \
            -textvariable $this-upper_threshold
        pack $w.upper_threshold.label $w.upper_threshold.entry -side left
        pack $w.upper_threshold

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
        
        frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top 

    }

}
