
catch {rename Isosurface ""}

package require Iwidgets 3.0

itcl_class Yarden_Visualization_Isosurface {
    inherit Module

    constructor {config} {
	set name Isosurface
	set_defaults
    }
    method set_defaults {} {
	global $this-isoval_min $this-isoval_max 
	global $this-visibility $this-value $this-scan
	global $this-bbox
	global $this-cutoff_depth 
	global $this-reduce
	global $this-all
	global $this-update
	global $this-rebuild
	global $this-min_size
	global $this-poll
	global $this-prev-isoval 
	global $this-alg
	global $this-show-span
	global $this-span-width
	global $this-span-height
	global $this-spanid
	global $this-span-region-set
	global $this-span-region-x0
	global $this-span-region-y0
	global $this-span-region-x1
	global $this-span-region-y1
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b

	set $this-isoval_min 0
	set $this-isoval_max 4095
	set $this-visiblilty 0
	set $this-value 1
	set $this-scan 1
	set $this-bbox 1
	set $this-reduce 1
	set $this-all 0
	set $this-update 0
	set $this-rebuild 0
	set $this-min_size 1
	set $this-poll 0
	set $this-prev-isoval 0
	set $this-alg 0
	set $this-show-span 0
	set $this-span-width   512
	set $this-span-height  512
	set $this-spanid 0
	set $this-span-region-set 0
	set $this-clr-r 0.6
	set $this-clr-g 0.114
	set $this-clr-b 0.15
    }

    method ui {} {
	global $this-isoval_min $this-isoval_max 
	global $this-cutoff_depth $this-bbox
	global $this-reduce $this-all
	global $this-update
	global $this-rebuild
	global $this-alg

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}      

	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	
	#  Info

	iwidgets::Labeledframe $w.f.info -labelpos nw -labeltext "Info"
	set info [$w.f.info childsite]
	
	label $info.type -text type 
	label $info.gen -text generation

	pack $info.type $info.gen -side left
	pack $w.f.info -side top -anchor w

	#  Options

	iwidgets::Labeledframe $w.f.opt -labelpos nw -labeltext "Options"
	set opt [$w.f.opt childsite]
	
	iwidgets::combobox $opt.update -labeltext "Update:" \
	    -selectioncommand "$this update-type $opt.update"
	
	$opt.update insert list end Manual Auto
	$opt.update selection set 0

	checkbutton $opt.hash -text "Hash" -relief flat \
	    -variable $this-hash
	checkbutton $opt.emit -text "Emit Surface" -relief flat \
	    -variable $this-emit

	pack $opt.update $opt.emit $opt.hash -side top -anchor w
	
	
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]
	frame $opt.col -relief ridge -borderwidth 4 -height 0.7c -width 0.7c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	button $opt.b -text "Set Color" -command "$this raiseColor $opt.col"
	pack $opt.b $opt.col -side left -fill x -padx 5 -expand 1
	pack $w.f.opt -side top -anchor w

	#  Isosurface
	iwidgets::Labeledframe $w.f.iso -labelpos nw -labeltext "Isovalue"
 	set iso [$w.f.iso childsite]

 	button $iso.extract -text "Extract" -relief raised -command $n

	scale $iso.isoval -label "Iso Value:" \
	    -variable $this-isoval \
	    -from [set $this-isoval_min] -to [set $this-isoval_max] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -digits 5 \
	    -resolution 0.1 \
	    -command "$this change_isoval"

	trace variable $this-isoval_min w "$this change_isoval_min"
	trace variable $this-isoval_max w "$this change_isoval_max"

	checkbutton $iso.show -text "Show Span Space"  \
	    -command "$this show-span $w.shell" -variable $this-show-span
	
# 	pack $opt.show -side top -anchor w

	pack $iso.extract -side top -anchor w
	pack $iso.isoval -side top -fill x
	pack $iso.show -side top -anchor w

	pack $w.f.iso -side top -anchor w -fill x
	
	iwidgets::shell $w.shell -modality none -title "Span Space" \
	    -height 512 -width 512
	
	image create photo $this-spanspace -file spanspace.ppm
	
	set width [set $this-span-width]
	set height [set $this-span-height]

	set span [canvas [$w.shell childsite].span \
		      -width $width -height $height]
	pack $span -anchor nw
	set $this-spanid $span

	$span create image 0 0 -image $this-spanspace -anchor nw -tags $this-bg

	set width2 [expr $width / 2]
	set height2 [expr $height / 2]
	$span create line 0 $height2 $width2 $height2 $width2 0 \
	    -tags $this-l1 -fill white -width 1

	$span create rectangle 0 0 0 0 -tags $this-r1 -width 1 -outline yellow 
	
	$span addtag $this-r1 above $this-l1

 	bind $span <ButtonPress-1> "$this press-1 $span %x %y"
 	bind $span <B1-Motion> "$this motion-1 $span %x %y"

 	bind $span <ButtonPress-2> "$this press-2 $span %x %y"
 	bind $span <B2-Motion> "$this motion-2 $span %x %y"

	bind $span <ButtonPress-3> "$this press-3 $span"

	#  Methods
	iwidgets::Labeledframe $w.f.method -labelpos nw -labeltext "Methods"
	set mf [$w.f.method childsite]
	
	iwidgets::tabnotebook  $mf.tabs -raiseselect true 
	#-fill both
	pack $mf.tabs -side top

	#  Methods: MC

	set alg [$mf.tabs add -label "MC" -command "set $this-alg 0"]

	#  Methods: NOISE

	set alg [$mf.tabs add -label "NOISE" -command "set $this-alg 1"]
	

	#  Methods: SAGE

	set alg [$mf.tabs add -label "SAGE" -command "set $this-alg 2"]

	iwidgets::checkbox $alg.prune -labeltext "Prune"
	$alg.prune add value  -text "Value"  -variable $this-value
	$alg.prune add bbox   -text "BBox"   -variable $this-bbox
	$alg.prune add scan   -text "Scan"   -variable $this-scan
	$alg.prune add points -text "Points" -variable $this-reduce

	iwidgets::checkbox $alg.opt -labeltext "Options"
	$alg.opt add poll   -text "Poll"   -variable $this-poll
	$alg.opt add size   -text "Size"   -variable $this-min_size
 	$alg.opt add all    -text "All"    -variable $this-all

	pack $alg.prune $alg.opt -side left -anchor n
	
	set alg [$mf.tabs add -label "Opt"]

	iwidgets::radiobox $alg.orient -labeltext "Tabs position:" \
	    -command "$this orient $mf $alg" 

	$alg.orient add n -text "n"
	$alg.orient add w -text "w"
	$alg.orient add e -text "e"
	$alg.orient add s -text "s"

	$alg.orient select n


	pack $alg.orient -padx 4 -pady 4 -anchor w

	$mf.tabs view "SAGE"
	$mf.tabs configure -tabpos [$alg.orient get]

	
	pack $mf.tabs -side top

	pack $w.f.method -side top

    }
   
    method change_isoval { n } {
	global $this-isoval $this-prev-isoval
        global $this-update
	
	if { [set $this-update] == 1 
	     && [set $this-isoval] != [set $this-prev-isoval]} {
	    set $this-prev-isoval [set $this-isoval]
	    eval "$this-c needexecute"
	}
    }

    method change_isoval_min {n1 n2 op} {
 	set iso [.ui[modname].f.iso childsite].isoval
	global $iso
	global $this-isoval_min

	$iso configure -from [set $this-isoval_min]
	puts "change_min [set $this-isoval_min]"
    }
    
    method change_isoval_max {n1 n2 op} {
 	set iso [.ui[modname].f.iso childsite].isoval
	global $iso
	global $this-isoval_max

	$iso configure -to [set $this-isoval_max]
    }

	
    method orient { tab page { val 4 }} {
	global $page
	global $tab
	
	$tab.tabs configure -tabpos [$page.orient get]
    }
    
    method update-type { w } {
	global $w
	global $this-update
	
	if { [$w getcurselection] == "Manual" } {
	    set $this-update 0
	} else {
	    set $this-update 1
	}
    }

    method show-span { w } {
	global $w
	global $this-show-span
	
	if { [set $this-show-span] == 1 } {
	    $w activate
	} else {
	    $w deactivate
	}
    }

    method press-1 { win x y } {
 	set iso [.ui[modname].f.iso childsite].isoval
	global $iso
	global $this-isoval
	global $this-isoval_max
	global $this-isoval_min
	global $this-span-width

	set width [set $this-span-width]
	set max  [set $this-isoval_max]
	set min  [set $this-isoval_min]
	set factor [expr $x * [expr $max - $min]]
	set v [expr  $min + [expr $factor / $width]]
	set $this-isoval $v

	$this span-cross $win $x
    }
	

    method motion-1 { win x y } {
 	set iso [.ui[modname].f.iso childsite].isoval
	global $iso
	global $this-isoval
	global $this-isoval_max
	global $this-isoval_min
	global $this-span-width

	set width [set $this-span-width]
	set max  [set $this-isoval_max]
	set min  [set $this-isoval_min]
	set factor [expr $x * [expr $max - $min]]
	set v [expr  $min + [expr $factor / $width]]
	set $this-isoval $v

	$this span-cross $win $x
    }

    method press-2 { win x y } {
	global $this-spanid
	set span [set $this-spanid]
	
	global $this-span-region-x0
	global $this-span-region-y0
	set $this-span-region-x0 $x
	set $this-span-region-y0 $y
	set $this-span-region-x1 $x
	set $this-span-region-y1 $y
	
	$span delete $this-r1
	$span create rectangle $x $y $x $y -tags $this-r1 -width 1 -outline yellow
	set $this-span-region-set 1
    }
    
    method motion-2 { win x y } {
	global $this-spanid
	set span [set $this-spanid]
	
	global $this-span-region-x0
	global $this-span-region-y0
	global $this-span-region-x1
	global $this-span-region-y1

	set px [set $this-span-region-x0]
	set py [set $this-span-region-y0]
	set $this-span-region-x1 $x
	set $this-span-region-y1 $y

	$win coords $this-r1 $px $py $x $y 
    }

    method press-3 { span } {
	global $span
	global $this-span-region-set
	global $this-span-region-x0
	global $this-span-region-y0
	global $this-span-region-x1
	global $this-span-region-y1

	if { [set $this-span-region-set] == 1 } {
	    set $this-span-region-set 0
	    puts "Region off"
	    $span delete $this-r1
#	    $span itemconfigure $this-r1 state hidden
	} else {
	    set $this-span-region-set 1
	    $span create rectangle \
		[set $this-span-region-x0] \
		[set $this-span-region-y0] \
		[set $this-span-region-x1] \
		[set $this-span-region-y1] \
		-tags $this-r1 -width 1 -outline yellow
	    puts "Region on"
#	    $span itemconfigure $this-r1 state normal
	}
    }

    method span-cross { win x } {
	global $win

	set y [expr 512 - $x]
	$win coords $this-l1 0 $y $x $y $x 0
    }

    method span-read-image { filename } {
	global $this-spanid

	set span [set $this-spanid]

	puts "span delete"
	$span delete $this-bg
	puts "image delete"
 	image delete $this-spanspace
	puts "image create $filename"
 	image create photo $this-spanspace -file $filename
	puts "span create"
	$span create image 0 0 -image $this-spanspace -anchor nw -tags $this-bg
	puts "span lower"
	$span lower $this-bg $this-l1  
	puts "done"
    }

    method raiseColor { col } {
	set w .ui[modname]
	if {[winfo exists $w.color]} {
	    raise $w.color
	    return;
	} else {
	    toplevel $w.color
	    global $this-clr
	    makeColorPicker $w.color $this-clr "$this setColor $col" \
		    "destroy $w.color"
	}
    }
    method setColor { col } {
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]

	$col config -background [format #%04x%04x%04x $ir $ig $ib]
    }
} 


