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
#  itk::NeighborhoodConnectedImageFilter
#############################################################


 itcl_class Insight_Filters_NeighborhoodConnectedImageFilter {
    inherit Module
    constructor {config} {
         set name NeighborhoodConnectedImageFilter

         global $this-replace_value
         global $this-lower_threshold
         global $this-upper_threshold
         global $this-dimension

         set_defaults
    }

    method set_defaults {} {

         set $this-radius 1
         set $this-seed_point 0
         set $this-replace_value 255.0
         set $this-lower_threshold 150
         set $this-upper_threshold 180
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

        frame $w.radius -relief groove -borderwidth 2
        pack $w.radius -padx 2 -pady 2 -side top -expand yes

        if {[set $this-dimension] == 0} {
            label $w.radius.label -text "Module must Execute to determine dimensions to build GUI for radius."
            pack $w.radius.label
       } else {
            init_radius_dimensions
       }

        frame $w.seed_point -relief groove -borderwidth 2
        pack $w.seed_point -padx 2 -pady 2 -side top -expand yes

        if {[set $this-dimension] == 0} {
            label $w.seed_point.label -text "Module must Execute to determine dimensions to build GUI for seed_point."
            pack $w.seed_point.label
       } else {
            init_seed_point_dimensions
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

        frame $w.upper_threshold
        label $w.upper_threshold.label -text "upper_threshold"
        entry $w.upper_threshold.entry \
            -textvariable $this-upper_threshold
        pack $w.upper_threshold.label $w.upper_threshold.entry -side left
        pack $w.upper_threshold
        
        frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top 

    }

    method clear_radius_gui {} {
        set w .ui[modname]

        for {set i 0} {$i < [set $this-dimension]} {incr i} {

            # destroy widget for each dimension
            if {[winfo exists $w.radius.radius$i]} {
		destroy $w.radius.radius$i
            }
        }

        # destroy label explaining need to execute
        if {[winfo exists $w.radius.label]} {
 		destroy $w.radius.label
        }
     }

    method clear_seed_point_gui {} {
        set w .ui[modname]

        for {set i 0} {$i < [set $this-dimension]} {incr i} {

            # destroy widget for each dimension
            if {[winfo exists $w.seed_point.seed_point$i]} {
		destroy $w.seed_point.seed_point$i
            }
        }

        # destroy label explaining need to execute
        if {[winfo exists $w.seed_point.label]} {
 		destroy $w.seed_point.label
        }
     }

    method init_radius_dimensions {} {
     	set w .ui[modname]
        if {[winfo exists $w]} {

            # destroy label explaining need to execute in case
            # it wasn't previously destroyed
	    if {[winfo exists $w.radius.label]} {
	       destroy $w.radius.label
            }

	    # pack new widgets for each dimension
            label $w.radius.label -text "radius (by dimension):"
            pack $w.radius.label -side top -padx 5 -pady 5 -anchor n
	    global $this-dimension

            for	{set i 0} {$i < [set $this-dimension]} {incr i} {
		if {! [winfo exists $w.radius.radius$i]} {
		    # create widget for this dimension
                    global $this-radius$i



                    if {[set $this-dimension] != 0 && [info exists $this-radius$i]} {
                      set $this-radius$i [set $this-radius$i]
                    } else {
                      set $this-radius$i 1
                    }

        frame $w.radius.radius$i
        label $w.radius.radius$i.label -text "radius in $i"
        entry $w.radius.radius$i.entry \
            -textvariable $this-radius$i
        pack $w.radius.radius$i.label $w.radius.radius$i.entry -side left
        pack $w.radius.radius$i

                }
            }
        }
    }

    method init_seed_point_dimensions {} {
     	set w .ui[modname]
        if {[winfo exists $w]} {

            # destroy label explaining need to execute in case
            # it wasn't previously destroyed
	    if {[winfo exists $w.seed_point.label]} {
	       destroy $w.seed_point.label
            }

	    # pack new widgets for each dimension
            label $w.seed_point.label -text "seed_point (by dimension):"
            pack $w.seed_point.label -side top -padx 5 -pady 5 -anchor n
	    global $this-dimension

            for	{set i 0} {$i < [set $this-dimension]} {incr i} {
		if {! [winfo exists $w.seed_point.seed_point$i]} {
		    # create widget for this dimension
                    global $this-seed_point$i



                    if {[set $this-dimension] != 0 && [info exists $this-seed_point$i]} {
                      set $this-seed_point$i [set $this-seed_point$i]
                    } else {
                      set $this-seed_point$i 0
                    }

        frame $w.seed_point.seed_point$i
        label $w.seed_point.seed_point$i.label -text "seed_point in $i"
        entry $w.seed_point.seed_point$i.entry \
            -textvariable $this-seed_point$i
        pack $w.seed_point.seed_point$i.label $w.seed_point.seed_point$i.entry -side left
        pack $w.seed_point.seed_point$i

                }
            }
        }
    }

}
