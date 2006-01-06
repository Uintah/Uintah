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


itcl_class SCIRun_Visualization_GenStandardColorMaps { 
    inherit Module 
    protected exposed
    protected colorMaps
    protected colorMap
    protected alphaMap
    protected curX
    protected curY
    protected selected
    constructor {config} { 
        set name GenStandardColorMaps 
        set_defaults 
	buildColorMaps
    } 
    
    method set_defaults {} { 
	global $this-faux
	global $this-gamma
	global $this-mapType
	global $this-mapName
	global $this-reverse
	global $this-resolution
	global $this-realres
	global $this-minRes
	global $this-nodeList
	global $this-positionList
	global $this-width
	global $this-height
        set $this-faux 0
	set $this-gamma 0
	set $this-mapType 3
	set $this-mapName "Rainbow"
	set $this-reverse 0
	set $this-resolution 256
	set $this-realres 256
	set $this-minRes 2
	set exposed 0
	set colorMap {}
	set selected -1
	set $this-nodeList {}
	set $this-positionList {}
	set $this-width 1
	set $this-height 1

	trace variable $this-mapType w "$this lookupOldIndex"
    }   
    
    method buildColorMaps {} {
	set colorMaps {
	    { "Gray" { { 0 0 0 } { 255 255 255 } } }
	    { "Rainbow" {
		{0 0 255} {0 52 255}
		{1 80 255} {3 105 255}
		{5 132 255} {9 157 243}
		{11 177 213} {15 193 182}
		{21 210 152} {30 225 126}
		{42 237 102} {60 248 82}
		{87 255 62} {116 255 49}
		{148 252 37} {178 243 27}
		{201 233 19} {220 220 14}
		{236 206 10} {247 185 8}
		{253 171 5} {255 151 3}
		{255 130 2} {255 112 1}
		{255 94 0} {255 76 0}
		{255 55 0} {255 0 0}}}
	    { "Old Rainbow" {
		{ 0 0 255}   { 0 102 255}
		{ 0 204 255}  { 0 255 204}
		{ 0 255 102}  { 0 255 0}
		{ 102 255 0}  { 204 255 0}
		{ 255 234 0}  { 255 204 0}
		{ 255 102 0}  { 255 0 0}}}
	    { "Darkhue" {
		{ 0  0  0 }  { 0 28 39 }
		{ 0 30 55 }  { 0 15 74 }
		{ 1  0 76 }  { 28  0 84 }
		{ 32  0 85 }  { 57  1 92 }
		{ 108  0 114 }  { 135  0 105 }
		{ 158  1 72 }  { 177  1 39 }
		{ 220  10 10 }  { 229 30  1 }
		{ 246 72  1 }  { 255 175 36 }
		{ 255 231 68 }  { 251 255 121 }
		{ 239 253 174 }}}
	    { "Lighthue" {
		{ 64  64  64 }  { 64 80 84 }
		{ 64 79 92 }  { 64 72 111 }
		{ 64  64 102 }  { 80 64 108 }
		{ 80 64 108 }  { 92  64 110 }
		{ 118  64 121 }  { 131  64 116 }
		{ 133  64 100 }  { 152  64 84 }
		{ 174  69 69 }  { 179 79  64 }
		{ 189 100  64 }  { 192 152 82 }
		{ 192 179 98 }  { 189 192 124 }
		{ 184 191 151 }}}
	    { "Blackbody" {
		{0 0 0}   {52 0 0}
		{102 2 0}   {153 18 0}
		{200 41 0}   {230 71 0}
		{255 120 0}   {255 163 20}
		{255 204 55}   {255 228 80}
		{255 247 120}   {255 255 180}
		{255 255 255}}}
	    { "Don" {
		{   0  90 255 }    {  51 104 255 }
		{ 103 117 255 }    { 166 131 245 }
		{ 181 130 216 }    { 192 129 186 }
		{ 197 128 172 }    { 230 126  98 }
		{ 240 126  49 }    { 255 133   0 }}}
	    { "BP Seismic" { { 0 0 255 } { 255 255 255} { 255 0 0 } } }
	    { "Dark Gray" {
		{   0  0  0 }    {  0 0 0 }
		{ 128 128 128 } { 255 255 255 }}}
	    { "Red Tint" { { 20 0 0 } { 255 235 235 } } }
	    { "Orange Tint" { { 20 10 0 } { 255 245 235 } } }
	    { "Yellow Tint" { { 20 20 0 } { 255 255 235 } } }
	    { "Green Tint" { { 0 20 0 } { 235 255 235 } } }
	    { "Cyan Tint" { { 0 20 20 } { 235 255 255 } } }
	    { "Blue Tint" { { 0 0 20 } { 235 235 255 } } }
	    { "Purple Tint" { { 10 0 20 } { 245 235 255 } } }
	}
    }
    
    method getMaps {} {
	set maps {}
	for {set i 0} { $i < [llength $colorMaps]} {incr i} {
	    lappend maps [list [lindex [lindex $colorMaps $i] 0] $i]
	}
	return [list $maps]
    }
    method ui {} { 
	global $this-minRes
	global $this-resolution
	global $this-realres
	
	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    return
	} 
	
	set type ""
	
	toplevel $w 
	wm minsize $w 200 50 
	
	frame $w.f -relief flat -borderwidth 2
	pack $w.f -side top -expand yes -fill x 
	
	frame $w.f.f1 -relief sunken -height 40  -borderwidth 2 
	pack $w.f.f1 -side right -padx 2 -pady 2 -expand yes -fill x


	canvas $w.f.f1.canvas -bg "#ffffff" -height 40 
	pack $w.f.f1.canvas -anchor w -expand yes -fill x

	TooltipMultiline $w.f.f1.canvas \
	    "The red line represents the alpha value.  Use the left mouse button to add a\n" \
	    "node for editing the line or to move an existing node.  You can use the\n" \
	    "right mouse button to delete a node.  Alpha defaults to 0.5."

	label $w.l0 -text "Click above to adjust alpha."
	pack $w.l0 -anchor c
	
	frame $w.f3 -relief groove -borderwidth 2
	pack $w.f3 -side top -anchor c -expand yes -fill x -padx 2
	scale $w.f3.s -orient horizontal -from -1 -to 1 -showvalue true \
	    -label "Shift" -variable $this-gamma -resolution 0.01 -tickinterval 1
	pack $w.f3.s -expand yes -fill x -padx 2
	
	Tooltip $w.f3.s "Skews the color map to the left or right."

	scale $w.f3.s2 -from [set $this-minRes] -to 256 -state normal \
		-orient horizontal  -variable $this-resolution -label "Resolution"
	pack $w.f3.s2 -expand yes -fill x -pady 2 -padx 2

	Tooltip $w.f3.s2 "Sets the number of unique colors used in the color map."
	
	bind $w.f3.s2 <ButtonRelease> \
	    "$this setres; $this update; $this-c needexecute"

	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -padx 2 -pady 2 -expand yes -fill both
	
	make_labeled_radio $w.f2.types "ColorMaps" "$this change" \
	    "split" $this-mapName [getColorMapNames]
	    
	pack $w.f2.types -expand yes -fill both

	frame $w.f4 -relief groove -borderwidth 2
	pack $w.f4 -padx 2 -pady 2 -expand yes -fill x

	checkbutton $w.f4.faux -text "Opacity Modulation (Faux Shading)" -relief flat \
            -variable $this-faux -onvalue 1 -offvalue 0 \
            -anchor w -command "$this-c needexecute"
        pack $w.f4.faux -side top -fill x -padx 4
	Tooltip $w.f4.faux "Modulates color components based on the given opacity curve."

	checkbutton $w.f4.reverse -text "Reverse the colormap" -relief flat \
            -variable $this-reverse -onvalue 1 -offvalue 0 \
            -anchor w -command "$this change"
        pack $w.f4.reverse -side top -fill x -padx 4
	Tooltip $w.f4.reverse "Reverse the colormap (not the alpha)"

	bind $w.f.f1.canvas <Expose> "$this canvasExpose"
	bind $w.f.f1.canvas <Button-1> "$this selectNode %x %y"
	bind $w.f.f1.canvas <B1-Motion> "$this moveNode %x %y"
	bind $w.f.f1.canvas <Button-3> "$this deleteNode %x %y"
	bind $w.f.f1.canvas <ButtonRelease> "$this update; $this-c needexecute"
	bind $w.f3.s <ButtonRelease> "$this change"
	$this update
	
	set cw [winfo width $w.f.f1.canvas]

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
   
    method setres {} {
	global $this-realres
	set $this-realres [set $this-resolution]
    }

    method change {} {
	$this update
	$this-c needexecute
    }

    method selectNode { x y } {
	set w .ui[modname]
	set c $w.f.f1.canvas
	set curX $x
	set curY $y
	
	set selected [$c find withtag current]
	set index [lsearch [set $this-nodeList] $selected]
	if { $index == -1 } {
	    makeNode $x $y
	    set index [lsearch [set $this-nodeList] $selected]
	} 
	set loc [$c coords $selected]
	if { $loc != "" } {
	    set curX [expr ([lindex $loc 0]+[lindex $loc 2])*0.5]
	    set curY [expr ([lindex $loc 1]+[lindex $loc 3])*0.5]
	}
    }
	

    method makeNode { x y } {
	set w .ui[modname]
	set c $w.f.f1.canvas
	set new [$c create oval [expr $x - 5] [expr $y - 5] \
		     [expr $x+5] [expr $y+5] -outline white \
		     -fill black -tags node]
	set selected $new
	$this nodeInsert $new [list $x $y] $x
	$this drawLines
    }

    method drawLines { } {
	global $this-nodeList
	global $this-positionList
	set w .ui[modname]
	set canvas $w.f.f1.canvas
	$canvas delete line
	set cw [winfo width $canvas]
	set ch [winfo height $canvas]
	set x 0
	set y [expr $ch/2]
	for { set i 0 } { $i < [llength [set $this-nodeList]]} {incr i} {
	    set p [lindex [set $this-positionList] $i]
	    $canvas create line $x $y [lindex $p 0] [lindex $p 1] \
		-tags line -fill red
	    set x [lindex $p 0]
	    set y [lindex $p 1]
	}
	$canvas create line $x $y $cw [expr $ch/2]  -tags line -fill red
	$canvas raise node line
    }

    method drawNodes { } {
	global $this-nodeList
	global $this-positionList
	set w .ui[modname]
	set c $w.f.f1.canvas
	set cw [winfo width $c]
	set ch [winfo height $c]
	for {set i 0} { $i < [llength [set $this-nodeList]] } {incr i} {
	    set x [lindex [lindex [set $this-positionList] $i] 0 ]
	    set y [lindex [lindex [set $this-positionList] $i] 1 ]
	    if { $x < 0 } {
		set x 0
	    } elseif { $x > $cw } {
		set x $cw 
	    }

	    if { $y < 0 } {
		set y 0
	    } elseif { $y > $ch } {
		set y $ch 
	    }
	    set new [$c create oval [expr $x - 5] [expr $y - 5] \
		     [expr $x+5] [expr $y+5] -outline white \
		     -fill black -tags node]
	    set $this-nodeList [lreplace [set $this-nodeList] $i $i $new]
	}
    }

    method nodeInsert { n p x} {
	global $this-nodeList
	global $this-positionList
	set index 0
	for { set i 0 } { $i < [llength [set $this-nodeList]] } { incr i } { 
	    if { $x < [lindex [lindex [set $this-positionList] $i] 0 ] } {
		break;
	    } else {
		incr index
	    }
	}
	set $this-nodeList [linsert [set $this-nodeList] $index $n]
	set $this-positionList [linsert [set $this-positionList] $index $p]
    }
    
    method moveNode { x y } {
	global $this-nodeList
	global $this-positionList
	set w .ui[modname]
	set c $w.f.f1.canvas
	set cw [winfo width $c]
	set ch [winfo height $c]
	if { $curX + $x-$curX < 0 } { set x 0 }
	if { $curX + $x-$curX > $cw } { set x $cw}
	if { $curY + $y-$curY < 0 } { set y 0 }
	if { $curY + $y-$curY > $ch } { set y $ch }
	set i [lsearch  [set $this-nodeList] $selected]
	if { $i != -1 } {
	    set l [lindex [set $this-nodeList] $i]
	    $c move $l [expr $x-$curX] [expr $y-$curY]
	    set curX $x
	    set curY $y
	    set $this-nodeList [lreplace [set $this-nodeList] $i $i]
	    set $this-positionList [lreplace [set $this-positionList] $i $i]
	    nodeInsert $l [list $x $y] $x
	    $this drawLines
	}
    }

    method lreverse { stuff } {
	set size [llength $stuff]
	set result {}
	for {set i 0}  {$i < $size} {incr i} {
	    set result [concat [list [lindex $stuff $i]] $result]
	}
	return $result
    }

    method findByName_aux { cname } {
	set size [llength $colorMaps]
	for {set i 0}  {$i < $size} {incr i} {
	    set cname1 [lindex [lindex $colorMaps $i] 0]
	    if {$cname == $cname1} { return $i }
	}
	return 0
    }

    method findByName { cname } {
	set index [findByName_aux $cname]
	set color [lindex [lindex $colorMaps $index] 1]
	if {[set $this-reverse]} { set color [lreverse $color] }
	return $color
    }

    method getColorMapNames {} {
	set size [llength $colorMaps]
	set result {}
	for {set i 0}  {$i < $size} {incr i} {
	    set cname [lindex [lindex $colorMaps $i] 0]
	    set result [concat $result [list [list $cname $cname]]]
	}
	return $result
    }
	
    method lookupOldIndex {a b c} {
	global $this-mapType
	set index [set $this-mapType]
	# Old name, new name, reverse?  Note that the order these are
	# listed is important because mapType is an index into this
	# list.
	set remap {
	    { "Gray" "Gray" 0 }
	    { "Inverse Gray" "Gray" 1 }
	    { "Rainbow" "Old Rainbow" 1 }
	    { "Inverse Rainbow " "Old Rainbow" 0 }
	    { "Darkhue" "Darkhue" 0 }
	    { "Inverse Darkhue" "Darkhue" 1 }
	    { "Lighthue" "Lighthue" 0 }
	    { "Blackbody" "Blackbody" 0 }
	    { "Inverse Blackbody" "Blackbody" 1 }
	    { "Don" "Don" 0 }
	    { "Dark Gray" "Dark Gray" 0 }
	    { "Red Tint" "Red Tint" 0 }
	    { "Orange Tint" "Orange Tint" 0 }
	    { "Yellow Tint" "Yellow Tint" 0 }
	    { "Green Tint" "Green Tint" 0 }
	    { "Blue Tint" "Blue Tint" 0 }
	    { "Purple Tint" "Purple Tint" 0 }
	    { "BP Seismic" "BP Seismic" 0}
           }
	set entry [lindex $remap $index]
	set $this-mapName [lindex $entry 1]
	set $this-reverse [lindex $entry 2]
    }

    method deleteNode { x y } {
	global $this-nodeList
	global $this-positionList
	set w .ui[modname]
	set c $w.f.f1.canvas
	set l [$c find withtag current]
	set i [lsearch  [set $this-nodeList] $l]
	if { $i != -1 } {
	    $c delete current
	    set $this-nodeList [lreplace [set $this-nodeList] $i $i]
	    set $this-positionList [lreplace [set $this-positionList] $i $i]
	    $this drawLines
	}
    }
    method update { } {
	$this SetColorMap
	$this redraw
	set selected -1
    }
    
    method getColorMapString { } {
	set colors [findByName [set $this-mapName]]
	set csize [llength $colors]
	set scolors [join $colors]

	global $this-positionList
	global $this-width
	global $this-height
	set cw [set $this-width]
	set ch [set $this-height]
	set alphas [join [set $this-positionList]]
	set asize [llength [set $this-positionList]]

        return "$csize $scolors $asize $cw $ch $alphas"
    }

    method close {} {
	set w .ui[modname]
	set exposed 0
	destroy $w
    }
    
    method canvasExpose {} {
	set w .ui[modname]
	
	if { [winfo viewable $w.f.f1.canvas] } { 
	    if { $exposed } {
		return
	    } else {
		set exposed 1
		$this drawNodes
		$this drawLines
		$this redraw
	    } 
	} else {
	    return
	}
    }
    
    method redraw {} {
	global $this-width
	global $this-height
	set w .ui[modname]
	
	if {![winfo exists $w]} { 
	    return
	} 

	set n [llength $colorMap]
	set canvas $w.f.f1.canvas
	$canvas delete map
	set cw [winfo width $canvas]
	set $this-width $cw
	set ch [winfo height $canvas]
	set $this-height $ch
	set dx [expr $cw/double($n)] 
	set x 0
	for {set i 0} {$i < $n} {incr i 1} {
	    set color [lindex $colorMap $i]
	    set r [lindex $color 0]
	    set g [lindex $color 1]
	    set b [lindex $color 2]
	    set c [format "#%02x%02x%02x" $r $g $b]
	    set oldx $x
	    set x [expr ($i+1)*$dx]
	    $canvas create rectangle \
		    $oldx 0 $x $ch -fill $c -outline $c -tags map
	}
	set taglist [$canvas gettags all]
	set i [lsearch $taglist line]
	if { $i != -1 } {
	    $canvas lower map line
	}
    }

    method SetColorMap {} {
	global $this-resolution
	global $this-mapName
	global $this-realres
	set colorMap {}
	set map [findByName [set $this-mapName]]
	set currentMap {}
	set currentMap [$this makeNewMap $map]
	set n [llength $currentMap]
	if { [set $this-resolution] > [set $this-realres] } {
	      set $this-resolution [set $this-realres]
	}
	set m [set $this-resolution]

	set frac [expr ($n-1)/double($m-1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
		lappend color [$this getAlpha $i]
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
		lappend color [$this getAlpha $i]
	    } else {
		set index [expr int($i * $frac)]
		set t [expr ($i * $frac)-$index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
		lappend color [$this getAlpha $i]
	    }
	    lappend colorMap $color
	}
    }
    

    method makeNewMap { currentMap } {
	global $this-gamma
	global $this-r

	set res [set $this-realres]
	set newMap {}
	set m [expr int($res + abs( [set $this-gamma] )*(255 - $res))]
	set n [llength $currentMap]
	if { $m < $n } { set m $n }
	set frac [expr double($n-1)/double($m - 1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
	    } else {
		set index_double [$this modify [expr $i * $frac] [expr $n-1]]
		
		set index [expr int($index_double)]
		set t  [expr $index_double - $index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
	    }
	    lappend newMap $color
	}
	return $newMap
    }
    method modify {  i range } {
	global $this-gamma
	
	set val [expr $i/double($range)]
	set bp [expr tan( 1.570796327*(0.5 + [set $this-gamma]*0.49999))]
	set index [expr pow($val,$bp)]
	return $index*$range
    }
    method getAlpha { index } {
	global nodeList
	global $this-positionList
	global $this-width
	global $this-height

	set cw [set $this-width]
	set ch [set $this-height]
	set m [llength [set $this-nodeList]]
	set dx [expr $cw/double([set $this-resolution])] 
	set xpos [expr int($index*$dx) + 0.5 * $dx]
	set x 0.0
	set y $ch/2.0
	set i 0
	for {set i 0} {$i <= $m} {incr i 1} {
	    set newx 0
	    set newy $ch/2.0
	    if { $i != $m } {
		set newx [lindex [lindex [set $this-positionList] $i] 0]
		set newy [lindex [lindex [set $this-positionList] $i] 1]
	    } else {
		set newy $ch/2.0
		set newx $cw
	    }

	    if { $xpos < $newx } {
		set frac [expr ( $xpos - $x )/double( $newx - $x )]
		return [expr 1.0 -  (($y + $frac * ($newy - $y))/double($ch))]
	    }
	    set x $newx
	    set y $newy
	}
    }
}

