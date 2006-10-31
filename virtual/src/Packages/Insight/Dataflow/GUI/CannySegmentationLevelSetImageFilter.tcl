#
#   For more information, please see: http://software.sci.utah.edu
#
#   The MIT License
#
#   Copyright (c) 2004 Scientific Computing and Imaging Institute,
#   University of Utah.
#
#   
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
#  itk::CannySegmentationLevelSetImageFilter
#############################################################


 itcl_class Insight_Filters_CannySegmentationLevelSetImageFilter {
    inherit Module
    constructor {config} {
         set name CannySegmentationLevelSetImageFilter

         global $this-iterations
         global $this-reverse_expansion_direction
         global $this-max_rms_change
         global $this-threshold
         global $this-variance
         global $this-propagation_scaling
         global $this-advection_scaling
         global $this-curvature_scaling
         global $this-isovalue
         global $this-update_OutputImage
         global $this-update_iters_OutputImage
    	 global $this-reset_filter

         set_defaults
    }

    method set_defaults {} {

         set $this-iterations 10
         set $this-reverse_expansion_direction 0
         set $this-max_rms_change 0.001
         set $this-threshold 0.001
         set $this-variance 10
         set $this-propagation_scaling 1.0
         set $this-advection_scaling 1.0
         set $this-curvature_scaling 1.0
         set $this-isovalue 0.5
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


        frame $w.iterations
        label $w.iterations.label -text "iterations"
        entry $w.iterations.entry \
            -textvariable $this-iterations
        pack $w.iterations.label $w.iterations.entry -side left
        pack $w.iterations

        frame $w.reverse_expansion_direction
        checkbutton $w.reverse_expansion_direction.checkbutton \
           -text "reverse_expansion_direction" \
           -variable $this-reverse_expansion_direction
        pack $w.reverse_expansion_direction.checkbutton -side left
        pack $w.reverse_expansion_direction

        frame $w.max_rms_change
        label $w.max_rms_change.label -text "max_rms_change"
        entry $w.max_rms_change.entry \
            -textvariable $this-max_rms_change
        pack $w.max_rms_change.label $w.max_rms_change.entry -side left
        pack $w.max_rms_change

        frame $w.threshold
        label $w.threshold.label -text "threshold"
        entry $w.threshold.entry \
            -textvariable $this-threshold
        pack $w.threshold.label $w.threshold.entry -side left
        pack $w.threshold


        frame $w.variance
        scale $w.variance.scale -label  "variance" \
           -variable $this-variance \
           -from 0 -to 100 -orient horizontal 
        pack $w.variance.scale -side left
        pack $w.variance

        frame $w.propagation_scaling
        label $w.propagation_scaling.label -text "propagation_scaling"
        entry $w.propagation_scaling.entry \
            -textvariable $this-propagation_scaling
        pack $w.propagation_scaling.label $w.propagation_scaling.entry -side left
        pack $w.propagation_scaling

        frame $w.advection_scaling
        label $w.advection_scaling.label -text "advection_scaling"
        entry $w.advection_scaling.entry \
            -textvariable $this-advection_scaling
        pack $w.advection_scaling.label $w.advection_scaling.entry -side left
        pack $w.advection_scaling

        frame $w.curvature_scaling
        label $w.curvature_scaling.label -text "curvature_scaling"
        entry $w.curvature_scaling.entry \
            -textvariable $this-curvature_scaling
        pack $w.curvature_scaling.label $w.curvature_scaling.entry -side left
        pack $w.curvature_scaling

        frame $w.isovalue
        label $w.isovalue.label -text "isovalue"
        entry $w.isovalue.entry \
            -textvariable $this-isovalue
        pack $w.isovalue.label $w.isovalue.entry -side left
        pack $w.isovalue

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

}
