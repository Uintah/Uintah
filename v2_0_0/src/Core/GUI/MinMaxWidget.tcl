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