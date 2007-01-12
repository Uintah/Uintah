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
#  itk::ThresholdSegmentationLevelSetImageFilter
#############################################################


 itcl_class Insight_Filters_ThresholdSegmentationLevelSetImageFilter {
    inherit Module
    constructor {config} {
         set name ThresholdSegmentationLevelSetImageFilter

         global $this-lower_threshold
         global $this-upper_threshold
         global $this-curvature_scaling
         global $this-propagation_scaling
         global $this-edge_weight
         global $this-max_iterations
         global $this-max_rms_change
         global $this-reverse_expansion_direction
         global $this-isovalue
         global $this-smoothing_iterations
         global $this-smoothing_time_step
         global $this-smoothing_conductance
         global $this-update_OutputImage
         global $this-update_iters_OutputImage
	 global $this-reset_filter

         set_defaults
    }

    method set_defaults {} {

         set $this-lower_threshold 210
         set $this-upper_threshold 250
         set $this-curvature_scaling 1
         set $this-propagation_scaling 1
         set $this-edge_weight 1.0
         set $this-max_iterations 120
         set $this-max_rms_change 0.02
         set $this-reverse_expansion_direction 0
         set $this-isovalue 127.5
         set $this-smoothing_iterations 0
         set $this-smoothing_time_step 0.1
         set $this-smoothing_conductance 0.5
         set $this-update_OutputImage 0
         set $this-update_iters_OutputImage 10
	 set $this-reset_filter 0
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

        frame $w.curvature_scaling
        label $w.curvature_scaling.label -text "curvature_scaling"
        entry $w.curvature_scaling.entry \
            -textvariable $this-curvature_scaling
        pack $w.curvature_scaling.label $w.curvature_scaling.entry -side left
        pack $w.curvature_scaling

        frame $w.propagation_scaling
        label $w.propagation_scaling.label -text "propagation_scaling"
        entry $w.propagation_scaling.entry \
            -textvariable $this-propagation_scaling
        pack $w.propagation_scaling.label $w.propagation_scaling.entry -side left
        pack $w.propagation_scaling

        frame $w.edge_weight
        label $w.edge_weight.label -text "edge_weight"
        entry $w.edge_weight.entry \
            -textvariable $this-edge_weight
        pack $w.edge_weight.label $w.edge_weight.entry -side left
        pack $w.edge_weight

        frame $w.max_iterations
        label $w.max_iterations.label -text "max_iterations"
        entry $w.max_iterations.entry \
            -textvariable $this-max_iterations
        pack $w.max_iterations.label $w.max_iterations.entry -side left
        pack $w.max_iterations

        frame $w.max_rms_change
        label $w.max_rms_change.label -text "max_rms_change"
        entry $w.max_rms_change.entry \
            -textvariable $this-max_rms_change
        pack $w.max_rms_change.label $w.max_rms_change.entry -side left
        pack $w.max_rms_change

        frame $w.reverse_expansion_direction
        checkbutton $w.reverse_expansion_direction.checkbutton \
           -text "reverse_expansion_direction" \
           -variable $this-reverse_expansion_direction
        pack $w.reverse_expansion_direction.checkbutton -side left
        pack $w.reverse_expansion_direction

        frame $w.isovalue
        label $w.isovalue.label -text "isovalue"
        entry $w.isovalue.entry \
            -textvariable $this-isovalue
        pack $w.isovalue.label $w.isovalue.entry -side left
        pack $w.isovalue

        frame $w.smoothing_iterations
        label $w.smoothing_iterations.label -text "smoothing_iterations"
        entry $w.smoothing_iterations.entry \
            -textvariable $this-smoothing_iterations
        pack $w.smoothing_iterations.label $w.smoothing_iterations.entry -side left
        pack $w.smoothing_iterations

        frame $w.smoothing_time_step
        label $w.smoothing_time_step.label -text "smoothing_time_step"
        entry $w.smoothing_time_step.entry \
            -textvariable $this-smoothing_time_step
        pack $w.smoothing_time_step.label $w.smoothing_time_step.entry -side left
        pack $w.smoothing_time_step

        frame $w.smoothing_conductance
        label $w.smoothing_conductance.label -text "smoothing_conductance"
        entry $w.smoothing_conductance.entry \
            -textvariable $this-smoothing_conductance
        pack $w.smoothing_conductance.label $w.smoothing_conductance.entry -side left
        pack $w.smoothing_conductance

        frame $w.updatesOutputImage
        checkbutton $w.updatesOutputImage.do \
            -text "Send intermediate updates for OutputImage" \
            -variable $this-update_OutputImage
        pack $w.updatesOutputImage.do -side top -anchor w
        frame $w.updatesOutputImage.i
        pack $w.updatesOutputImage.i -side top -anchor nw
        label $w.updatesOutputImage.i.l -text "Send Intermediate Interval:"
        entry $w.updatesOutputImage.i.e -textvariable $this-update_iters_OutputImage -width 10
        pack $w.updatesOutputImage.i.l $w.updatesOutputImage.i.e -side left -anchor w -padx 2 -pady 2
        pack $w.updatesOutputImage        

	button $w.stop -text "Stop Segmentation" \
	    -command "$this-c stop_segmentation"
	pack $w.stop -side top -anchor n

	button $w.reset -text "Reset Filter" \
	    -command "set $this-reset_filter 1"
	pack $w.reset -side top -anchor n

        frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top 

    }

     method update_guivars_from_data_dictionary {pairs} {
	 # For each pair, set the corresponding guivar.
	 # If it doesn't exist, it won't hurt anything
	 
	 for {set i 0} {$i < [llength $pairs]} {incr i} {
	     set which [lindex $pairs $i]
	     set key [lindex $which 0]
	     set value [lindex $which 1]
	     global $this-$key
	     set $this-$key $value
	 }
	 
     }
}
