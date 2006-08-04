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
#  itk::IsolatedConnectedImageFilter
#############################################################


 itcl_class Insight_Filters_IsolatedConnectedImageFilter {
    inherit Module
    constructor {config} {
         set name IsolatedConnectedImageFilter

         global $this-replace_value
         global $this-lower_threshold
         global $this-upper_value_limit
         global $this-isolated_value_tolerance
         global $this-dimension

         set_defaults
    }

    method set_defaults {} {

         set $this-seed_point_1 0
         set $this-seed_point_2 0
         set $this-replace_value 255.0
         set $this-lower_threshold 150
         set $this-upper_value_limit 255
         set $this-isolated_value_tolerance 0
         set $this-dimension 0
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


        wm minsize $w 150 80

        frame $w.seed_point_1 -relief groove -borderwidth 2
        pack $w.seed_point_1 -padx 2 -pady 2 -side top -expand yes

        if {[set $this-dimension] == 0} {
            label $w.seed_point_1.label -text "Module must Execute to determine dimensions to build GUI for seed_point_1."
            pack $w.seed_point_1.label
       } else {
            init_seed_point_1_dimensions
       }

        frame $w.seed_point_2 -relief groove -borderwidth 2
        pack $w.seed_point_2 -padx 2 -pady 2 -side top -expand yes

        if {[set $this-dimension] == 0} {
            label $w.seed_point_2.label -text "Module must Execute to determine dimensions to build GUI for seed_point_2."
            pack $w.seed_point_2.label
       } else {
            init_seed_point_2_dimensions
       }

        frame $w.replace_value
        label $w.replace_value.label -text "replace_value"
        entry $w.replace_value.entry \
            -textvariable $this-replace_value
        pack $w.replace_value.label $w.replace_value.entry -side left
        pack $w.replace_value

        frame $w.lower_threshold
        label $w.lower_threshold.label -text "lower_threshold"
        entry $w.lower_threshold.entry \
            -textvariable $this-lower_threshold
        pack $w.lower_threshold.label $w.lower_threshold.entry -side left
        pack $w.lower_threshold

        frame $w.upper_value_limit
        label $w.upper_value_limit.label -text "upper_value_limit"
        entry $w.upper_value_limit.entry \
            -textvariable $this-upper_value_limit
        pack $w.upper_value_limit.label $w.upper_value_limit.entry -side left
        pack $w.upper_value_limit

        frame $w.isolated_value_tolerance
        label $w.isolated_value_tolerance.label -text "isolated_value_tolerance"
        entry $w.isolated_value_tolerance.entry \
            -textvariable $this-isolated_value_tolerance
        pack $w.isolated_value_tolerance.label $w.isolated_value_tolerance.entry -side left
        pack $w.isolated_value_tolerance
        
        frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top 

    }

    method clear_seed_point_1_gui {} {
        set w .ui[modname]

        for {set i 0} {$i < [set $this-dimension]} {incr i} {

            # destroy widget for each dimension
            if {[winfo exists $w.seed_point_1.seed_point_1$i]} {
		destroy $w.seed_point_1.seed_point_1$i
            }
        }

        # destroy label explaining need to execute
        if {[winfo exists $w.seed_point_1.label]} {
 		destroy $w.seed_point_1.label
        }
     }

    method clear_seed_point_2_gui {} {
        set w .ui[modname]

        for {set i 0} {$i < [set $this-dimension]} {incr i} {

            # destroy widget for each dimension
            if {[winfo exists $w.seed_point_2.seed_point_2$i]} {
		destroy $w.seed_point_2.seed_point_2$i
            }
        }

        # destroy label explaining need to execute
        if {[winfo exists $w.seed_point_2.label]} {
 		destroy $w.seed_point_2.label
        }
     }

    method init_seed_point_1_dimensions {} {
     	set w .ui[modname]
        if {[winfo exists $w]} {

            # destroy label explaining need to execute in case
            # it wasn't previously destroyed
	    if {[winfo exists $w.seed_point_1.label]} {
	       destroy $w.seed_point_1.label
            }

	    # pack new widgets for each dimension
            label $w.seed_point_1.label -text "seed_point_1 (by dimension):"
            pack $w.seed_point_1.label -side top -padx 5 -pady 5 -anchor n
	    global $this-dimension

            for	{set i 0} {$i < [set $this-dimension]} {incr i} {
		if {! [winfo exists $w.seed_point_1.seed_point_1$i]} {
		    # create widget for this dimension
                    global $this-seed_point_1$i



                    if {[set $this-dimension] != 0 && [info exists $this-seed_point_1$i]} {
                      set $this-seed_point_1$i [set $this-seed_point_1$i]
                    } else {
                      set $this-seed_point_1$i 0
                    }

        frame $w.seed_point_1.seed_point_1$i
        label $w.seed_point_1.seed_point_1$i.label -text "seed_point_1 in $i"
        entry $w.seed_point_1.seed_point_1$i.entry \
            -textvariable $this-seed_point_1$i
        pack $w.seed_point_1.seed_point_1$i.label $w.seed_point_1.seed_point_1$i.entry -side left
        pack $w.seed_point_1.seed_point_1$i

                }
            }
        }
    }

    method init_seed_point_2_dimensions {} {
     	set w .ui[modname]
        if {[winfo exists $w]} {

            # destroy label explaining need to execute in case
            # it wasn't previously destroyed
	    if {[winfo exists $w.seed_point_2.label]} {
	       destroy $w.seed_point_2.label
            }

	    # pack new widgets for each dimension
            label $w.seed_point_2.label -text "seed_point_2 (by dimension):"
            pack $w.seed_point_2.label -side top -padx 5 -pady 5 -anchor n
	    global $this-dimension

            for	{set i 0} {$i < [set $this-dimension]} {incr i} {
		if {! [winfo exists $w.seed_point_2.seed_point_2$i]} {
		    # create widget for this dimension
                    global $this-seed_point_2$i



                    if {[set $this-dimension] != 0 && [info exists $this-seed_point_2$i]} {
                      set $this-seed_point_2$i [set $this-seed_point_2$i]
                    } else {
                      set $this-seed_point_2$i 0
                    }

        frame $w.seed_point_2.seed_point_2$i
        label $w.seed_point_2.seed_point_2$i.label -text "seed_point_2 in $i"
        entry $w.seed_point_2.seed_point_2$i.entry \
            -textvariable $this-seed_point_2$i
        pack $w.seed_point_2.seed_point_2$i.label $w.seed_point_2.seed_point_2$i.entry -side left
        pack $w.seed_point_2.seed_point_2$i

                }
            }
        }
    }

}
