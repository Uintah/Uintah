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
