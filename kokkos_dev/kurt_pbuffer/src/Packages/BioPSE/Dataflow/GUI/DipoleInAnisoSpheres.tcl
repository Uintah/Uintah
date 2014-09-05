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

##
 #  DipoleInAnisoSpheres.tcl:
 #
 #  Author: Sascha Moehrs
 #
 ##

package require Iwidgets 3.0

catch {rename DipoleInAnisoSpheres ""}

itcl_class BioPSE_Forward_DipoleInAnisoSpheres {
    inherit Module
    constructor {config} {
        set name DipoleInAnisoSpheres
        set_defaults
    }
    method set_defaults {} {

		# accuracy of the series expansion / max expansion terms
		global $this-accuracy
		set $this-accuracy 0.00001
		global $this-expTerms
		set $this-expTerms 100

    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w
		# wm minsize $w 300 100

		frame $w.f 
		pack $w.f -padx 2 -pady 2 -expand 1 -fill x

		# accuracy / expansion terms
		iwidgets::labeledframe $w.f.a -labelpos "nw" -labeltext "series expansion"
		set ac [$w.f.a childsite]

		global $this-accuracy
		label $ac.la -text "accuracy: "
		entry $ac.ea -width 20 -textvariable $this-accuracy
		bind  $ac.ea <Return> "$this-c needexecute"
		grid  $ac.la -row 0 -column 0 -sticky e
		grid  $ac.ea -row 0 -column 1 -columnspan 2 -sticky "ew"
		
		global $this-expTerms
		label  $ac.le -text "expansion terms: "
		entry  $ac.ee -width 20 -textvariable $this-expTerms -state disabled
		# bind   $ac.ee <Return> "$this-c needexecute"
		grid   $ac.le -row 1 -column 0 -sticky e
		grid   $ac.ee -row 1 -column 1 -columnspan 2 -sticky "ew"

		grid columnconfigure . 1 -weight 1

		pack $w.f.a -side top -fill x -expand 1

    }
}
