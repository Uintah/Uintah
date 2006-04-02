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


itcl_class SCIRun_FieldsOther_ScalarFieldStats {
    inherit Module
    protected draw_graph_needed 0
    constructor {config} {
        set name ScalarFieldStats

        global $this-min
        global $this-max
        global $this-mean
        global $this-median
        global $this-sigma
        global $this-is_fixed
        global $this-nbuckets
	global $this-setdata
	global $this-nmin
	global $this-nmax
        global $this-args

        set_defaults
    }

    method set_defaults {} {
        set $this-min "?"
        set $this-max "?"
        set $this-mean "?"
        set $this-median "?"
        set $this-sigma "?"
        set $this-is_fixed 0
        set $this-nbuckets 256
	set $this-setdata 0
	set $this-nmin 0
	set $this-nmax 256
	set $this-args "?"
   }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        set n "$this-c needexecute "

        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Range"  -variable $this-is_fixed \
                -value 0 -command "$this autoRange"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Range"  -variable $this-is_fixed \
                -value 1 -command "$this fixedRange"
        pack $w.f2.b -side left

        frame $w.row1
        frame $w.row2
        frame $w.row3
        frame $w.row4

        pack $w.row1 $w.row2 $w.row3 $w.row4 \
            -side top -e y -f both -padx 5 
        
        label $w.row1.min_label -text "Min Value:  "
        entry $w.row1.min_value -textvariable $this-min
        label $w.row1.max_label -text ",  Max Value:  "
        entry $w.row1.max_value -textvariable $this-max
        pack $w.row1.min_label $w.row1.min_value \
            $w.row1.max_label $w.row1.max_value -side left

        bind $w.row1.min_value <Return> $n
        bind $w.row1.max_value <Return> $n

        label $w.row2.mean_label -text "Mean:  "
        label $w.row2.mean_value -textvariable $this-mean
        pack $w.row2.mean_label $w.row2.mean_value -side left

        label $w.row3.median_label -text "Median:  "
        label $w.row3.median_value -textvariable $this-median
        pack $w.row3.median_label $w.row3.median_value -side left

        label $w.row4.sigma_label -text "Standard Deviation:  "
        label $w.row4.sigma_value -textvariable $this-sigma
        pack $w.row4.sigma_label $w.row4.sigma_value -side left

        blt::barchart $w.graph -title "Histogram" \
            -height [expr [set $this-nbuckets]*3/4.0] \
            -width [set $this-nbuckets] -plotbackground gray80 \
	    -barmode aligned
        pack $w.graph

        frame $w.size -relief flat
        pack $w.size -side top -expand yes -fill x
        label $w.size.l -text "Histogram Bins:  "
        entry $w.size.e -textvariable $this-nbuckets
        pack $w.size.l $w.size.e -side left -expand yes -pady 3

        bind $w.size.e <Return> "$this resize_graph; $n"

        makeSciButtonPanel $w $w $this
        moveToCursor $w

        if { [set $this-is_fixed] } {
            $w.f2.b select
            $this fixedRange
        } else {
            $w.f1.b select
            global $this-is_fixed
            set w .ui[modname]
            
            set $this-is_fixed 0
            
            set color "#505050"
            
            $w.row1.min_label configure -foreground $color
            $w.row1.min_value configure -state disabled -foreground $color
            $w.row1.max_label configure -foreground $color
            $w.row1.max_value configure -state disabled -foreground $color
        }

        if { $draw_graph_needed } {
            $this-c needexecute
        }

    }

    method resize_graph { } {
        global $this-nbuckets
        set w .ui[modname]
        
        $w.graph configure -width [set $this-nbuckets]
        $w.graph configure -height [expr [set $this-nbuckets]*3/4.0]
    }

    method autoRange { } {
        global $this-is_fixed
        set w .ui[modname]
        
        set $this-is_fixed 0

        set color "#505050"

        $w.row1.min_label configure -foreground $color
        $w.row1.min_value configure -state disabled -foreground $color
        $w.row1.max_label configure -foreground $color
        $w.row1.max_value configure -state disabled -foreground $color
        $this-c needexecute     

    }

    method fixedRange { } {
        global $this-is_fixed
        set w .ui[modname]

        set $this-is_fixed 1


        $w.row1.min_label configure -foreground black
        $w.row1.min_value configure -state normal -foreground black
        $w.row1.max_label configure -foreground black
        $w.row1.max_value configure -state normal -foreground black
        
    }

    method tick_format { w val } {
	set s [format "%2.2e" $val]
	return $s
    }
    method graph_data { nmin nmax args } {
        global $this-min
        global $this-min
        global $this-setdata
        
        set w .ui[modname]
        if {[winfo exists $w.graph] != 1} {
            set draw_graph_needed 1
	    if {[set $this-setdata] == 1} {
		set $this-nmin $nmin
		set $this-nmax $nmax
		set $this-args $args
	    }
            return
        } else {
            set draw_graph_needed 0
        }
        
        if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
            $w.graph axis configure y -logscale yes
        } else {
            $w.graph axis configure y -logscale no
        }

        set min [set $this-min]
        set max [set $this-max]

        set xvector {}
        set yvector {}
        set yvector [concat $yvector $args]
        set frac [expr 1.0/double([llength $yvector]-1)]
	set bw [expr ($max - $min)/double([llength $yvector] -1)]
	$w.graph configure -barwidth $bw

	set interval [expr ($max - $min)/3.0]
	$w.graph axis configure x -min $min \
	    -max $max -command "$this tick_format" \
	    -subdivisions 2 -loose 1 -stepsize $interval

        for {set i 0} { $i < [llength $yvector] } {incr i} {
  	    set val  [expr $min + $i*$frac*($max-$min)]
            lappend xvector $val
        }
        

         if { [$w.graph element exists data] == 1 } {
             $w.graph element delete data
         }

	$w.graph element create data -label {} -xdata $xvector -ydata $yvector
	$w.graph element configure data -fg blue -relief flat -stipple ""
    }
    method clear_data { } {
	set w .ui[modname]
        if {[winfo exists $w.graph]} {
	    if { [$w.graph element exists data] == 1 } {
		$w.graph element delete data
	    }
	}
    }
}


