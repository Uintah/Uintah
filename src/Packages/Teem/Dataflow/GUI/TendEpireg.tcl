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

#    File   : TendEpireg.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendEpireg ""}

itcl_class Teem_Tend_TendEpireg {
    inherit Module
    constructor {config} {
        set name TendEpireg
        set_defaults
    }
    method set_defaults {} {
        global $this-gradient_list
        set $this-gradient_list ""

        global $this-reference
        set $this-reference "-1"

        global $this-blur_x
        set $this-blur_x 1.0

        global $this-blur_y
        set $this-blur_y 2.0

	global $this-use-default-threshold
	set $this-use-default-threshold 1

        global $this-threshold
        set $this-threshold 0.0

        global this-cc_analysis
        set $this-cc_analysis 1

        global $this-fitting
        set $this-fitting 0.70

        global $this-kernel
        set $this-kernel "cubicCR"

        global $this-sigma
        set $this-sigma 0.0

	global $this-extent
	set $this-extent 0.5
    }

    method update_text {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
            set $this-gradient_list [$w.f.options.gradient_list get 1.0 end]
        }
    }

    method send_text {} {
	$this update_text
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	option add *textBackground white	
	iwidgets::scrolledtext $w.f.options.gradient_list \
	    -vscrollmode dynamic -labeltext "List of gradients. example: (one gradient per line) 0.5645 0.32324 0.4432454"

	catch {$w.f.options.gradient_list insert end [set $this-gradient_list]}

        pack $w.f.options.gradient_list -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.reference \
	    -labeltext "reference:" -textvariable $this-reference
        pack $w.f.options.reference -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_x -labeltext "blur_x:" \
	    -textvariable $this-blur_x
        pack $w.f.options.blur_x -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_y -labeltext "blur_y:" \
	    -textvariable $this-blur_y
        pack $w.f.options.blur_y -side top -expand yes -fill x
	checkbutton $w.f.options.usedefaultthreshold -text \
	    "Use Default Threshold" -variable $this-use-default-threshold
	pack $w.f.options.usedefaultthreshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" \
	    -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.cc_analysis \
	    -labeltext "cc_analysis:" -textvariable $this-cc_analysis
        pack $w.f.options.cc_analysis -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.fitting -labeltext "fitting:" \
	    -textvariable $this-fitting
        pack $w.f.options.fitting -side top -expand yes -fill x

	make_labeled_radio $w.f.options.kernel "Kernel:" "" \
		top $this-kernel \
		{{"Box" box} \
		{"Tent" tent} \
		{"Cubic (Catmull-Rom)" cubicCR} \
		{"Cubic (B-spline)" cubicBS} \
		{"Quartic" quartic} \
		{"Windowed Sinc" hann} \
		{"Gaussian" gaussian}}

        pack $w.f.options.kernel -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.sigma -labeltext "sigma:" \
	    -textvariable $this-sigma
        pack $w.f.options.sigma -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.extent -labeltext "extent:" \
	    -textvariable $this-extent
        pack $w.f.options.extent -side top -expand yes -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
