#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

#
#  MRITissueClassifier.tcl
#  Written by:
#   McKay Davis
#   March 2004
#   Copyright (C) 2004 Scientific Computing and Imaging Institute
#

catch {rename Teem_Segmentation_MRITissueClassifier ""}

itcl_class Teem_Segmentation_MRITissueClassifier {
    inherit Module
    constructor {config} {
        set name MRITissueClassifier
        set_defaults
    }
    method set_defaults {} {
        setGlobal $this-maxIter 1000
	setGlobal $this-minChange 0.1
	setGlobal $this-top 1
	setGlobal $this-anterior 1
	setGlobal $this-eyesVisible 0
	setGlobal $this-pixelDim 1.0
	setGlobal $this-sliceThickness 3.0
    }

    method ui {} {
	global $this-maxIter $this-minChange $this-top

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }      
        toplevel $w
	set W $w

	set w $W.iter
        frame $w
        label $w.l -text "Max Iterations"
        pack $w.l -side left -expand 1 -fill x
        entry $w.e -textvariable $this-maxIter
        pack $w.e -side right -expand 0 -fill none
	pack $w -side top -expand 1 -fill x


	set w $W.minchange
        frame $w
        label $w.l -text "Minimum Change"
        pack $w.l -side left -expand 1 -fill x
        entry $w.e -textvariable $this-minChange
        pack $w.e -side right -expand 0 -fill none
	pack $w -side top -expand 1 -fill x

	set w $W.pixeldim
        frame $w
        label $w.l -text "Pixel Dimensions"
        pack $w.l -side left -expand 1 -fill x
        entry $w.e -textvariable $this-pixelDim
        pack $w.e -side right -expand 0 -fill none
	pack $w -side top -expand 1 -fill x

	set w $W.slicethick
        frame $w
        label $w.l -text "Slice Thickness (mm)"
        pack $w.l -side left -expand 1 -fill x
        entry $w.e -textvariable $this-sliceThickness
        pack $w.e -side right -expand 0 -fill none
	pack $w -side top -expand 1 -fill x

	set w $W.top
        frame $w
	checkbutton $w.but -text Top -variable $this-top
	pack $w.but -side left -expand 1 -fill x
	pack $w -side top -expand 1 -fill x

	set w $W.anterior
        frame $w
	checkbutton $w.but -text Anterior -variable $this-anterior
	pack $w.but -side left -expand 1 -fill x
	pack $w -side top -expand 1 -fill x

	set w $W.eyes
        frame $w
	checkbutton $w.but -text Eyes -variable $this-eyesVisible
	pack $w.but -side left -expand 1 -fill x
	pack $w -side top -expand 1 -fill x
	
	makeSciButtonPanel $W $W $this
	moveToCursor $W

    }
}
