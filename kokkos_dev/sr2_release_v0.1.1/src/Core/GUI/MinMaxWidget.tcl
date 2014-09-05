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

#    File   : MinMaxWidget.tcl
#    Author : Martin Cole
#    Date   : Wed Feb 12 09:42:29 2003


# proc mine_changed {wmin wmax vmin vmax} {
#     if {[set $vmax] < [set $vmin]} {
# 	set $vmax [set $vmin]
# 	$wmin configure -to [set $vmin]	
#     }
#     min_changed $wmax $vmin $vmax 
# }

# proc maxe_changed {wmin wmax vmin vmax} {
#     if {[set $vmin] > [set $vmax]} {
# 	$wmax configure -from [set $vmax]
#     }
#     max_changed $wmin $vmin $vmax 
# }

proc min_changed {wmax vmin vmax} {
    if {[set $vmax] < [set $vmin]} {
	set $vmax [set $vmin]
    }
}

proc max_changed {wmin vmin vmax} {
    if {[set $vmin] > [set $vmax]} {
	set $vmin [set $vmax]
    }
}
proc set_scale_max_value {parent value} {
    set lf [$parent childsite]
    $lf.min configure -to $value
    $lf.max configure -to $value
}

proc min_max_widget {f nm min max maxval} {
    iwidgets::labeledframe $f -labelpos nw -labeltext $nm
    set lf [$f childsite]

    label $lf.minlab -text "Min:"
    scale $lf.min -orient horizontal -variable $min \
	-from 0 -to [set $maxval] -width 14 -showvalue 0

    entry $lf.mine -textvariable $min -width 6 
    bind $lf.mine <Return> "min_changed $lf.max $min $max"

    label $lf.maxlab -text "Max:"
    scale $lf.max -orient horizontal -variable $max \
	-from 0  -to [set $maxval] -width 14 -showvalue 0
    entry $lf.maxe -textvariable $max -width 6 
    bind $lf.maxe <Return> "max_changed $lf.min $min $max"

    bind $lf.min <ButtonRelease-1> \
	"min_changed $lf.max $min $max"
    bind $lf.max <ButtonRelease-1> \
	"max_changed $lf.min $min $max"

    pack $lf.minlab $lf.mine $lf.min -side left
    pack $lf.maxlab $lf.maxe $lf.max -side left
    pack $f
}