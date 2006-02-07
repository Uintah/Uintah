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

#    File   : TendPoint.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendPoint ""}

itcl_class Teem_Tend_TendPoint {
    inherit Module
    constructor {config} {
        set name TendPoint
        set_defaults
    }
    method set_defaults {} {
	global $this-firstwidth
	set $this-firstwidth 17

        global $this-x
        set $this-x 0

        global $this-y
        set $this-y 0

        global $this-z
        set $this-z 0

	global $this-confidence
	set $this-confidence "- - - -"

	global $this-tensor1
	global $this-tensor2
	global $this-tensor3
	global $this-tensor4
	global $this-tensor5
	global $this-tensor6
	set $this-tensor1 "- - - -"
	set $this-tensor2 "- - - -"
	set $this-tensor3 "- - - -"
	set $this-tensor4 "- - - -"
	set $this-tensor5 "- - - -"
	set $this-tensor6 "- - - -"

	global $this-eval1
	global $this-eval2
	global $this-eval3
	set $this-eval1 "- - - -"
	set $this-eval2 "- - - -"
	set $this-eval3 "- - - -"

	global $this-evec1
	global $this-evec2
	global $this-evec3
	global $this-evec4
	global $this-evec5
	global $this-evec6
	global $this-evec7
	global $this-evec8
	global $this-evec9
	
	set $this-evec1 "- - - -"
	set $this-evec2 "- - - -"
	set $this-evec3 "- - - -"
	set $this-evec4 "- - - -"
	set $this-evec5 "- - - -"
	set $this-evec6 "- - - -"
	set $this-evec7 "- - - -"
	set $this-evec8 "- - - -"
	set $this-evec9 "- - - -"

	global $this-angle
	set $this-angle "- - - -"
	global $this-axis1
	global $this-axis2
	global $this-axis3
	set $this-axis1 "- - - -"
	set $this-axis2 "- - - -"
	set $this-axis3 "- - - -"

	global $this-mat1
	global $this-mat2
	global $this-mat3
	global $this-mat4
	global $this-mat5
	global $this-mat6
	global $this-mat7
	global $this-mat8
	global $this-mat9
	set $this-mat1 "- - - -"
	set $this-mat2 "- - - -"
	set $this-mat3 "- - - -"
	set $this-mat4 "- - - -"
	set $this-mat5 "- - - -"
	set $this-mat6 "- - - -"
	set $this-mat7 "- - - -"
	set $this-mat8 "- - - -"
	set $this-mat9 "- - - -"

	global $this-cl1
	global $this-cp1
	global $this-ca1
	global $this-cs1
	global $this-ct1
	global $this-cl2
	global $this-cp2
	global $this-ca2
	global $this-cs2
	global $this-ct2
	global $this-ra
	global $this-fa
	global $this-vf
	global $this-b
	global $this-q
	global $this-r
	global $this-s
	global $this-skew
	global $this-th
	global $this-cz
	global $this-det
	global $this-tr

	set $this-cl1 "- - - -"
	set $this-cp1 "- - - -"
	set $this-ca1 "- - - -"
	set $this-cs1 "- - - -"
	set $this-ct1 "- - - -"
	set $this-cl2 "- - - -"
	set $this-cp2 "- - - -"
	set $this-ca2 "- - - -"
	set $this-cs2 "- - - -" 
	set $this-ct2 "- - - -"
	set $this-ra "- - - -"
	set $this-fa "- - - -"
	set $this-vf "- - - -"
	set $this-b "- - - -"
	set $this-q "- - - -"
	set $this-r "- - - -"
	set $this-s "- - - -"
	set $this-skew "- - - -"
	set $this-th "- - - -"
	set $this-cz "- - - -"
	set $this-det "- - - -"
	set  $this-tr "- - - -"	
    }
    
    method build_info_widgets {w} {
	labelpair $w.conf "Confidence" $this-confidence

	iwidgets::labeledframe $w.tensor \
	    -labeltext "Tensor" -labelpos nw
	pack $w.tensor -side top -anchor nw -expand yes
	
	set tensor [$w.tensor childsite]
	frame $tensor.a 
	pack $tensor.a -side top -expand yes
	label $tensor.a.1 -textvar $this-tensor1
	label $tensor.a.2 -textvar $this-tensor2
	label $tensor.a.3 -textvar $this-tensor3
	pack $tensor.a.1 $tensor.a.2 $tensor.a.3 -side left

	frame $tensor.b 
	pack $tensor.b -side top -expand yes
	label $tensor.b.1 -textvar $this-tensor4
	label $tensor.b.2 -textvar $this-tensor5
	label $tensor.b.3 -textvar $this-tensor6
	pack $tensor.b.1 $tensor.b.2 $tensor.b.3 -side left

	# for {set i 1} {$i < 7} {incr i} {
	    # labelpair $w.tensor$i "Tensor\[[expr $i - 1]\]" $this-tensor$i
	# }

	iwidgets::labeledframe $w.eigen \
	    -labeltext "Eigenvalues" -labelpos nw
	pack $w.eigen -side top -anchor nw

	set eigen [$w.eigen childsite]

	frame $eigen.val
	pack $eigen.val -side top -expand yes
	label $eigen.val.1 -textvar $this-eval1
	label $eigen.val.2 -textvar $this-eval2
	label $eigen.val.3 -textvar $this-eval3
	pack $eigen.val.1 $eigen.val.2 $eigen.val.3 -side left

	# for {set i 1} {$i < 4} {incr i} {
	    # labelpair $w.eval$i "Eigenvalue\[[expr $i - 1]\]" $this-eval$i
	# }

	iwidgets::labeledframe $w.eigenv \
	    -labeltext "Eigenvectors" -labelpos nw
	pack $w.eigenv -side top -anchor nw

	set eigenv [$w.eigenv childsite]

	frame $eigenv.val1
	pack $eigenv.val1 -side top -anchor nw -expand yes
	label $eigenv.val1.1 -textvar $this-evec1
	label $eigenv.val1.2 -textvar $this-evec2
	label $eigenv.val1.3 -textvar $this-evec3
	pack $eigenv.val1.1 $eigenv.val1.2 $eigenv.val1.3 -side left

	frame $eigenv.val2
	pack $eigenv.val2 -side top -anchor nw -expand yes
	label $eigenv.val2.1 -textvar $this-evec4
	label $eigenv.val2.2 -textvar $this-evec5
	label $eigenv.val2.3 -textvar $this-evec6
	pack $eigenv.val2.1 $eigenv.val2.2 $eigenv.val2.3 -side left

	frame $eigenv.val3
	pack $eigenv.val3 -side top -anchor nw -expand yes
	label $eigenv.val3.1 -textvar $this-evec7
	label $eigenv.val3.2 -textvar $this-evec8
	label $eigenv.val3.3 -textvar $this-evec9
	pack $eigenv.val3.1 $eigenv.val3.2 $eigenv.val3.3 -side left	


	# for {set i 1} {$i < 10} {incr i} {
	    # labelpair $w.evec$i "Eigenvector\[[expr $i - 1]\]" $this-evec$i
	# }

	labelpair $w.rot "Eigenvector\nrotation" $this-angle

	frame $w.rot2 
	pack $w.rot2 -side top -anchor nw -padx 5
	label $w.rot2.l1 -text "Rotation around X" -width [set $this-firstwidth] \
		      -anchor w -just left
	label $w.rot2.colon  -text ":" -width 2 -anchor w -just left 
	label $w.rot2.l2 -textvar $this-axis1 -width 20 -anchor w -just left \
		-fore darkred

	pack $w.rot2.l1 $w.rot2.colon -side left
	pack $w.rot2.l2 -side left -fill x -expand yes

	frame $w.rot3
	pack $w.rot3 -side top -anchor nw -padx 5
	label $w.rot3.l1 -text "Rotation around Y" -width [set $this-firstwidth] \
		      -anchor w -just left
	label $w.rot3.colon  -text ":" -width 2 -anchor w -just left 
	label $w.rot3.l2 -textvar $this-axis2 -width 20 -anchor w -just left \
		-fore darkred

	pack $w.rot3.l1 $w.rot3.colon -side left
	pack $w.rot3.l2 -side left -fill x -expand yes
	
	frame $w.rot4
	pack $w.rot4 -side top -anchor nw -padx 5
	label $w.rot4.l1 -text "Rotation around Z" -width [set $this-firstwidth] \
		      -anchor w -just left
	label $w.rot4.colon  -text ":" -width 2 -anchor w -just left 
	label $w.rot4.l2 -textvar $this-axis3 -width 20 -anchor w -just left \
		-fore darkred

	pack $w.rot4.l1 $w.rot4.colon -side left
	pack $w.rot4.l2 -side left -fill x -expand yes




	# for {set i 1} {$i < 4} {incr i} {
	    # labelpair $w.axis$i "Axis\[[expr $i - 1]\]" $this-axis$i
	# }
	iwidgets::labeledframe $w.aniso \
	    -labeltext "Anisotropies" -labelpos nw
	pack $w.aniso -side top -anchor nw -expand yes

	set ani [$w.aniso childsite]

	frame $ani.a
	frame $ani.b
	pack $ani.a $ani.b -side left -anchor nw

	labelpair_small $ani.a.cl1 "Cl1" $this-cl1
	labelpair_small $ani.a.cp1 "Cp1" $this-cp1
	labelpair_small $ani.a.ca1 "Ca1" $this-ca1
	labelpair_small $ani.a.cs1 "Cs1" $this-cs1
	labelpair_small $ani.a.ct1 "Ct1" $this-ct1
	labelpair_small $ani.a.cl2 "Cl2" $this-cl2
	labelpair_small $ani.a.cp2 "Cp2" $this-cp2
	labelpair_small $ani.a.ca2 "Ca2" $this-ca2
	labelpair_small $ani.a.cs2 "Cs2" $this-cs2
	labelpair_small $ani.a.ct2 "Ct2" $this-ct2
	labelpair_small $ani.a.ra "RA" $this-ra
	labelpair_small $ani.b.fa "FA" $this-fa
	labelpair_small $ani.b.vf "VF" $this-vf
	labelpair_small $ani.b.b "B" $this-b
	labelpair_small $ani.b.q "Q" $this-q
	labelpair_small $ani.b.r "R" $this-r
	labelpair_small $ani.b.s "S" $this-s
	labelpair_small $ani.b.skew "Skew" $this-skew
	labelpair_small $ani.b.th "Th" $this-th
	labelpair_small $ani.b.cz "Cz" $this-cz
	labelpair_small $ani.b.det "Det" $this-det
	labelpair_small $ani.b.tr "Tr" $this-tr
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	iwidgets::labeledframe $w.f.options \
	    -labeltext "Location: " \
	    -labelpos nw
	pack $w.f.options -side top -anchor nw -expand yes

	set location [$w.f.options childsite]
        iwidgets::entryfield $location.x -labeltext "X:" \
	    -textvariable $this-x -width 8

        iwidgets::entryfield $location.y -labeltext "Y:" \
	    -textvariable $this-y -width 8

        iwidgets::entryfield $location.z -labeltext "Z:" \
	    -textvariable $this-z -width 8
        pack $location.x $location.y $location.z -side left -expand yes -fill x

	frame $w.f.info
	pack $w.f.info -side top -expand yes

	build_info_widgets $w.f.info

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }

    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -anchor nw -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 20 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon -side left
	pack $win.l2 -side left -fill x -expand yes
    } 

    method labelpair_small { win text1 text2 } {
	frame $win 
	pack $win -side top -anchor nw -padx 5
	label $win.l1 -text $text1 -width [expr [set $this-firstwidth] / 2] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 20 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon -side left
	pack $win.l2 -side left -fill x -expand yes
    } 

}
