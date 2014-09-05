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
catch {rename EditColorMap2D ""}

itcl_class SCIRun_Visualization_EditColorMap2D {
    inherit Module
    protected highlighted
    protected frames
    protected close
    protected swatches
    protected swatchpath
    constructor {config} {
        set name EditColorMap2D
	set highlighted -1
	set frames ""
        set_defaults
	set pix [netedit getenv SCIRUN_SRCDIR]/pixmaps/powerapp-close.ppm
	set close [image create photo -file $pix]
	set swatches 0
        set swatchpath "[netedit getenv HOME]/SCIRun/Colormaps"
	if { ![validDir $swatchpath] } {
	    file mkdir $swatchpath
	}
	if { ![validDir $swatchpath] } {
	    set swatchpath ""
	}
    }
    
    method set_defaults {} {
        setGlobal $this-faux 0
        setGlobal $this-histo 0.5
        setGlobal $this-num-entries 2
        setGlobal $this-filename "MyTransferFunction"
        setGlobal $this-filetype Binary
	setGlobal $this-selected_widget -1
	setGlobal $this-selected_object -1
	global $this-selected_widget
	trace variable $this-selected_widget w "$this highlight_entry"
	global $this-marker
        trace variable $this-marker w "$this unpickle"
    }
    
    method unpickle { args } {
        $this-c unpickle
        global $this-marker
        trace vdelete $this-marker w "$this unpickle"
    }
    
    method raise_color {button color module_command} {
        global $color
        set windowname .ui[modname]_color
        if {[winfo exists $windowname]} {
	    destroy $windowname
	}
	# makeColorPicker now creates the $window.color toplevel.
	makeColorPicker $windowname $color \
	    "$this set_color $button $color \{$module_command\}" \
	    "destroy $windowname"
    }
    
    method set_color { button color { module_command "" } } {
	upvar \#0 $color-r r $color-g g $color-b b
	# format the r,g,b colors into a hexadecimal string representation
	set colstr [format \#%04x%04x%04x [expr int($r * 65535)] \
			[expr int($g * 65535)] [expr int($b * 65535)]]
	set button [join [lrange [split $button .] end-1 end] .]
	foreach frame $frames {
	    foreach but [info commands ${frame}*${button}] {
		$but config -background $colstr -activebackground $colstr
	    }
	}
	if { [string length $module_command] } {
	    eval $this-c $module_command
	}
    }
    
    method add_frame { frame } {
	if { [lsearch $frames $frame] == -1 } {
	    lappend frames $frame
	}
	create_entries
    }

    method select_widget { entry } {
	setGlobal $this-selected_widget $entry
	setGlobal $this-selected_object 1
	$this-c select_widget
    }

    method create_entries { } {
	foreach frame $frames {
	    update_widget_frame $frame
	}
    }

    # dont pass default argument to un-highlight all entires
    method highlight_entry { args } {
	upvar \#0 $this-selected_widget entry
	set force [string equal [lindex $args 0] 1]
	if { !$force && $highlighted == $entry } return;
	global Color
	foreach f $frames {
	    color_entry $f $highlighted $Color(Basecolor)
	    color_entry $f $entry LightYellow
	}
	set highlighted $entry
    }

    method color_entry { frame entry { color LightBlue3 } } {
	if { $entry == -1 } return
	set e $frame.e-$entry
	if { ![winfo exists $e] } return
	$e.name configure -background $color
	$e.shade configure -background $color -activebackground $color
	$e.on configure -background $color  -activebackground $color
	$e.fill configure -background $color 
	$e configure -background $color
    }
	

    method update_widget_frame { frame } {
	if { ![winfo exists $frame] } return
	set parent [join [lrange [split $frame .] 0 end-1] .]
	bind $parent <ButtonPress> "$this select_widget -1"
	# Create the new variables and entries if needed.
	upvar $this-num-entries num
	set force 0
	for {set i 0} {$i < $num} {incr i} {

	    initGlobal $this-name-$i default-$i
	    initGlobal $this-$i-color-r [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-g [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-b [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-a 0.7

	    set e $frame.e-$i
	    if { ![winfo exists $e] } {
		set force 1
		frame $e -width 500 -relief sunken -bd 1
		button $e.x -image $close -anchor nw -relief flat \
		    -command "$this select_widget $i; $this-c deletewidget"

		entry $e.name -textvariable $this-name-$i -width 13 -bd 0
		bind $e.name <ButtonPress> "+$this select_widget $i"
		button $e.color -width 4 \
		    -command "$this select_widget $i; $this raise_color $frame.e-$i.color $this-$i-color \"color $i\""

		checkbutton $e.shade -text "" -padx 15 -justify center \
		    -relief flat -variable $this-shadeType-$i \
		    -onvalue 1 -offvalue 0 -anchor w \
		    -command "$this select_widget $i; $this-c shade $i; "

		checkbutton $e.on -text "" -padx 15 -justify center \
		    -relief flat -variable $this-on-$i -onvalue 1 -offvalue 0 \
		    -anchor w -command "$this-c toggle $i"
		frame $e.fill -width 500
		bind $e.fill <ButtonPress> "$this select_widget $i"
		pack $e.x $e.name $e.color $e.shade $e.on $e.fill -side left \
		    -ipadx 0 -ipady 0 -padx 0 -pady 0 -fill y
		pack $e
	    }
	    set_color $e.color $this-$i-color
	}
	
	# Destroy all the left over entries from prior runs.
	while {[winfo exists $frame.e-$i]} {
	    destroy $frame.e-$i
	    incr i
	}
	highlight_entry $force
    }

    method file_save {} {
        set ws .ui[modname]-fbs
        if {[winfo exists $ws]} { 
	    SciRaise $ws
	    return 
        }
        # file types to appers in filter box
        set types { { {SCIRun 2D Transfer Function} {.cmap .cmap2} } }
        makeSaveFilebox \
	    -parent [toplevel $ws -class TkFDialog] \
	    -filevar $this-filename \
	    -setcmd "wm withdraw $ws" \
	    -command "$this-c save; wm withdraw $ws" \
	    -cancel "wm withdraw $ws" \
	    -title "Save SCIRun 2D Transfer Function" \
	    -filetypes $types \
	    -initialfile TransferFunc01 \
	    -initialdir ~/ \
	    -defaultextension .cmap2 \
	    -formatvar $this-filetype
	SciRaise $ws
    }
    
    method file_load {} {
        set wl .ui[modname]-fbl
        if {[winfo exists $wl]} {
	    SciRaise $wl
            return
        }
        # file types to appers in filter box
        set types { 
	    { {SCIRun 2D Transfer Function} {.cmap .cmap2} }
	    { {All Files} {.*} }
	}
        makeOpenFilebox \
	    -parent [toplevel $wl -class TkFDialog] \
	    -filevar $this-filename \
	    -command "$this-c load;  wm withdraw $wl" \
	    -cancel "wm withdraw $wl" \
	    -title "Open SCIRun 2D Transfer Function" \
	    -filetypes $types \
	    -initialdir ~/ \
	    -defaultextension .cmap2
	SciRaise $wl
    }

    method create_swatches {} {
        set swatches 0
	foreach file [glob -nocomplain ${swatchpath}/*.cmap2] {
	    $this add_swatch $file
	}
    }

    method add_swatch { filename } {
	if { ![validFile $filename] || ![validFile ${filename}.ppm] } return
        set w .ui[modname].swatchpicker
        if {![winfo exists $w]} return
	set col [expr $swatches % 4]
	set row [expr $swatches / 4]
	set f [$w childsite].swatchFrame$row
	if {$col == 0} {
	    frame $f
	    pack $f -side top -anchor nw
	}
	#Load in the image to diplay on the button
	image create photo img-$swatches -format ppm -file ${filename}.ppm
	button $f.swatch$swatches -image img-$swatches \
	    -command "$this swatch_load $filename"
	grid configure $f.swatch$swatches -row $row -col $col -sticky nw
	incr swatches
    }

    method swatch_delete {} {
	upvar \#0 $this-deleteSwatch delete
	if { ![file writable $delete] } return
        file delete $delete
	file delete ${delete}.ppm
        set f .ui[modname].swatchpicker
        if {![winfo exists $f]} return
	foreach element [winfo children [$f childsite]] {
	    destroy $element
	}
	create_swatches
    }

    method swatch_load { filename } {
	setGlobal $this-filename $filename
	setGlobal $this-deleteSwatch $filename
        $this-c load
    }

    method swatch_save {} {
	if { $swatchpath == "" } return
        set basename [clock format [clock seconds] -format {%Y%m%d_%H%M%S}]
        setGlobal $this-filename "${swatchpath}/${basename}.cmap2"
        $this-c save true
	add_swatch "${swatchpath}/${basename}.cmap2"
    }

    method label_widget_columns { frame } {
        frame $frame
        frame $frame.empty0 -width 21 -bd 0
        label $frame.name -text "Name" -width 13 -relief groove
        label $frame.color -text "Color" -width 4 -relief groove
        label $frame.shade -text "Solid" -width 7 -relief groove
        label $frame.onoff -text "On" -width 7 -relief groove
        label $frame.empty -text "" -width 3
        pack $frame.empty0 $frame.name $frame.color $frame.shade \
	    $frame.onoff $frame.empty -side left 
	return $frame
    }

    method ui {} {
        global $this-num-entries
        trace vdelete $this-num-entries w "$this unpickle"

        set w .ui[modname]
        if {[winfo exists $w]} {
	    SciRaise $w
            return
        }
        toplevel $w

	# create an OpenGL widget
	frame $w.glf -bd 2 -relief groove
	$this-c setgl $w.glf.gl
	bind_events $w.glf.gl
	pack $w.glf.gl -side top -padx 0 -pady 0 -expand 0
	pack $w.glf -expand 0

	set frame $w.topframe
        frame $frame
        pack $frame -padx 2 -pady 2 -fill x
	# histogram opacity
	scale $frame.shisto -variable $this-histo \
	    -from 0.0 -to 1.0 -label "Histogram Opacity" \
	    -showvalue true -resolution 0.001 \
	    -orient horizontal -command "$this-c redraw"
	pack $frame.shisto -side top -fill x -padx 4

	# faux shading
	frame $frame.f0 -relief groove -borderwidth 2
	pack $frame.f0 -padx 2 -pady 2 -fill x
	checkbutton $frame.f0.faux -text "Opacity Modulation (Faux Shading)" \
	    -relief flat -variable $this-faux -onvalue 1 -offvalue 0 \
	    -anchor w -command "$this-c redraw; $this-c needexecute"
	pack $frame.f0.faux -side top -fill x -padx 4

	# Scrollable frame areas for widget controls and swatches
        iwidgets::scrolledframe $w.widgets -hscrollmode none \
	    -vscrollmode static
        iwidgets::scrolledframe $w.swatchpicker -hscrollmode none \
	    -vscrollmode dynamic

	# W Controls
	set f $w.controls
        frame $f
        button $f.addtriangle -text "Add Triangle" \
            -command "$this-c addtriangle" -width 12
        button $f.addrectangle -text "Add Rectangle" \
            -command "$this-c addrectangle" -width 12
        button $f.delete -text Delete -command "$this-c deletewidget" -width 12
        button $f.undo -text Undo -command "$this-c undowidget" -width 12
        button $f.load -text Load -command "$this file_load" -width 12
        button $f.save -text Save -command "$this file_save" -width 12
        pack $f.addtriangle $f.addrectangle $f.delete $f.undo $f.load $f.save \
            -padx 8 -pady 4 -fill x -side left
	
	# Swatch Controls
	set f $w.swatchcontrol
	frame $f
	button $f.save -text QuickSave -command "$this swatch_save" -width 20
	button $f.del -text "Delete Swatch" \
	    -command "$this swatch_delete" -width 20
	pack $f.save $f.del -padx 8 -pady 4 -side left -fill x -expand 1
	
	# Pack 'em all up...
        pack [label_widget_columns $w.title]  -fill x -padx 2 -pady 2
        pack $w.widgets -side top -fill both -expand yes -padx 2
        pack $w.controls -fill x         
	pack $w.swatchpicker -side top -fill both -expand yes -padx 2
	pack $w.swatchcontrol -side top
	
	add_frame [$w.widgets childsite]
        create_entries
	create_swatches

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
            
    method bind_events {w} {
        bind $w <Destroy>		"$this-c destroygl"
        # every time the OpenGL widget is displayed, redraw it
        bind $w <Expose>		"$this-c redraw"
        bind $w <Configure>		"$this-c redraw"
        bind $w <ButtonPress-1>		"$this-c mouse push %x %y %b"
        bind $w <ButtonPress-2>		"$this-c mouse push %x %y %b"
        bind $w <ButtonPress-3>		"$this-c mouse push %x %y %b"
        bind $w <Motion>		"$this-c mouse motion %x %y %b"
        bind $w <Button1-Motion>	"$this-c mouse motion %x %y %b"
        bind $w <Button2-Motion>	"$this-c mouse motion %x %y %b"
        bind $w <Button3-Motion>	"$this-c mouse motion %x %y %b"
        bind $w <ButtonRelease-1>	"$this-c mouse release %x %y %b"
        bind $w <ButtonRelease-2>	"$this-c mouse release %x %y %b"
        bind $w <ButtonRelease-3>	"$this-c mouse release %x %y %b"

        # controls for pan and zoom and reset
        bind $w <Shift-ButtonPress-1>   "$this-c mouse x_late_start %x %y %b"
        bind $w <Shift-Button1-Motion>  "$this-c mouse x_late_motion %x %y %b"
        bind $w <Shift-ButtonRelease-1> "$this-c mouse x_late_end %x %y %b"
        bind $w <Shift-ButtonPress-2>   "$this-c mouse reset %x %y %b"
        bind $w <Shift-ButtonPress-3>   "$this-c mouse scale_start %x %y %b"
        bind $w <Shift-Button3-Motion>  "$this-c mouse scale_motion %x %y %b"
        bind $w <Shift-ButtonRelease-3> "$this-c mouse scale_end %x %y %b"
    }
}

