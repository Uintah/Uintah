
#
# For more information, please see: http://software.sci.utah.edu
#
# The MIT License
#
# Copyright (c) 2004 Scientific Computing and Imaging Institute,
# University of Utah.
#
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

#############################################################
# This is an automatically generated file for the
# itk::GeodesicActiveContourLevelSetImageFilter
#############################################################


 itcl_class Insight_Filters_GeodesicActiveContourLevelSetImageFilter {
    inherit Module
    constructor {config} {
         set name GeodesicActiveContourLevelSetImageFilter

         global $this-derivativeSigma
	 global $this-curvatureScaling
         global $this-propagationScaling
         global $this-advectionScaling
         global $this-max_iterations
         global $this-max_rms_change
         global $this-reverse_expansion_direction
         global $this-isovalue
         global $this-update_OutputImage
         global $this-update_iters_OutputImage
	 global $this-reset_filter

         set_defaults
    }

    method set_defaults {} {

         set $this-derivativeSigma 0
set $this-curvatureScaling 1.0
set $this-propagationScaling 1.0
set $this-advectionScaling 1.0
         set $this-max_iterations 120
         set $this-max_rms_change 0.02
         set $this-reverse_expansion_direction 0
         set $this-isovalue 127.5
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


        frame $w.derivativeSigma
        label $w.derivativeSigma.label -text "derivativeSigma"
        entry $w.derivativeSigma.entry \
            -textvariable $this-derivativeSigma
        pack $w.derivativeSigma.label $w.derivativeSigma.entry -side left
        pack $w.derivativeSigma

frame $w.curvatureScaling
        label $w.curvatureScaling.label -text "curvatureScaling"
        entry $w.curvatureScaling.entry \
            -textvariable $this-curvatureScaling
        pack $w.curvatureScaling.label $w.curvatureScaling.entry -side left
        pack $w.curvatureScaling

frame $w.propagationScaling
        label $w.propagationScaling.label -text "propagationScaling"
        entry $w.propagationScaling.entry \
            -textvariable $this-propagationScaling
        pack $w.propagationScaling.label $w.propagationScaling.entry -side left
        pack $w.propagationScaling

frame $w.advectionScaling
        label $w.advectionScaling.label -text "advectionScaling"
        entry $w.advectionScaling.entry \
            -textvariable $this-advectionScaling
        pack $w.advectionScaling.label $w.advectionScaling.entry -side left
        pack $w.advectionScaling
        
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
