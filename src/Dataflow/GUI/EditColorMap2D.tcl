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
    constructor {config} {
        set name EditColorMap2D
	set highlighted -1
	set frames ""
        set_defaults
    }
    
    method set_defaults {} {
        setGlobal $this-faux 0
        setGlobal $this-histo 0.5
        setGlobal $this-num-entries 2
        setGlobal $this-filename "MyTransferFunction"
        setGlobal $this-filetype Binary
        setGlobal $this-row 0
        setGlobal $this-col 0
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
	    "$this set_color $button $color $module_command" \
	    "destroy $windowname"
    }
    
    method set_color { button color { module_command "" } } {
	upvar \#0 $color-r r $color-g g $color-b b
	# format the r,g,b colors into a hexadecimal string representation
	set colstr [format \#%04x%04x%04x [expr int($r * 65535)] \
			[expr int($g * 65535)] [expr int($b * 65535)]]
	$button config -background $colstr -activebackground $colstr
	if { [string length $module_command] } {
	    $this-c $module_command
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
	$this-c resend_selection
	$this-c redraw true
    }

    method create_entries { } {
	foreach frame $frames {
	    update_widget_frame $frame
	}
    }

    # dont pass default argument to un-highlight all entires
    method highlight_entry { args } {
	upvar \#0 $this-selected_widget entry
	if { $highlighted == $entry } return;
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
	for {set i 0} {$i < $num} {incr i} {

	    initGlobal $this-name-$i default-$i
	    initGlobal $this-$i-color-r [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-g [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-b [expr rand()*0.75+0.25]
	    initGlobal $this-$i-color-a 0.7

	    set e $frame.e-$i
	    if { ![winfo exists $e] } {
		frame $e -width 500 -relief sunken -bd 1
		entry $e.name -textvariable $this-name-$i -width 14 -bd 1
		bind $e.name <ButtonPress> "+$this select_widget $i"
		button $e.color -width 4 \
		    -command "$this select_widget $i; $this raise_color $frame.e-$i.color $this-$i-color color_change-$i"

		checkbutton $e.shade -text "" -padx 19 -justify center \
		    -relief flat -variable $this-shadeType-$i \
		    -onvalue 1 -offvalue 0 -anchor w \
		    -command "$this-c shadewidget-$i; $this select_widget $i"

		checkbutton $e.on -text "" -padx 19 -justify center \
		    -relief flat -variable $this-on-$i -onvalue 1 -offvalue 0 \
		    -anchor w -command "$this-c toggleon-$i"
		frame $e.fill -width 500
		bind $e.fill <ButtonPress> "$this select_widget $i"
		pack $e.name $e.color $e.shade $e.on $e.fill -side left \
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
	highlight_entry
    }

    method file_save {} {
        global $this-filename
        set ws [format "%s-fbs" .ui[modname]]

        if {[winfo exists $ws]} { 
          if {[winfo ismapped $ws] == 1} {
            raise $ws
          } else {
            wm deiconify $ws
          }
          return 
        }

        toplevel $ws -class TkFDialog
        set initdir "~/"

        # place to put preferred data directory
        # it's used if $this-filename is empty
    
        #######################################################
        # to be modified for particular reader

        # extansion to append if no extension supplied by user
        set defext ".cmap2"

        # name to appear initially
        set defname "TransferFunc01"
        set title "Save Transfer Function"

        # file types to appers in filter box
        set types {
            {{All Files}       {.cmap .cmap2}   }
        }

        ######################################################

        makeSaveFilebox \
                -parent $ws \
                -filevar $this-filename \
                -setcmd "wm withdraw $ws" \
                -command "$this-c save; wm withdraw $ws" \
                -cancel "wm withdraw $ws" \
                -title $title \
                -filetypes $types \
                -initialfile $defname \
                -initialdir $initdir \
                -defaultextension $defext \
                -formatvar $this-filetype
                #-splitvar $this-split

    }
    
    method file_load {} {
        global $this-filename

        set wl [format "%s-fbl" .ui[modname]]
        if {[winfo exists $wl]} {
            if {[winfo ismapped $wl] == 1} {
                raise $wl
            } else {
                wm deiconify $wl
            }
            return
        }
        toplevel $wl -class TkFDialog
        set initdir "~/"
        #######################################################
        # to be modified for particular reader

        # extansion to append if no extension supplied by user
        set defext ".cmap2"
        set title "Open SCIRun Transfer Function file"

        # file types to appers in filter box
        set types {
            {{SCIRun Transfer Function}     {.cmap .cmap2}      }
            {{All Files} {.*}   }
        }

        ######################################################

        makeOpenFilebox \
                -parent $wl \
                -filevar $this-filename \
                -command "$this-c load;  wm withdraw $wl" \
                -cancel "wm withdraw $wl" \
                -title $title \
                -filetypes $types \
                -initialdir $initdir \
                -defaultextension $defext
    }

    method create_swatches {} {
        global $this-filename

        set path "[netedit getenv HOME]/SCIRun"
	if { ! [file isdirectory $path] } {
	    file mkdir $path
	}
        set curdir [pwd]
        cd $path
        set res [glob -nocomplain  *.ppm]
        cd $curdir
        set w .ui[modname]
        
        # use globals to keep track of where you are
        # when adding new buttons
        set row [set $this-row]
        set col [set $this-col]
        if {[winfo exists $w.swatchpicker]} {
          set f [$w.swatchpicker childsite]

            # Create row 0
            frame $f.swatchFrame$row
            pack $f.swatchFrame$row

            foreach r $res {
                # Every 4 buttons, create new row and reset col
                if {$col == 4} {
                    incr row
                    frame $f.swatchFrame$row
                    pack $f.swatchFrame$row -side top -anchor nw
                    set col 0
                }
                
                set num [lindex [split $r "."] 0]
                image create photo img-$r -format "ppm" -file $path/$num.ppm
                #Load in the image to diplay on the button and do that.
                button $f.swatchFrame$row.swatch$num -image img-$r -command "set $this-filename $path/$num.ppm.cmap2; $this swatch_load $num"
                grid configure $f.swatchFrame$row.swatch$num -row $row -col $col -sticky "nw"
                incr col
          }
        }
        set $this-row $row
        set $this-col $col
    }

    method update_swatches {file} {
        set row [set $this-row]
        set col [set $this-col]
 
        set w .ui[modname]
        if {[winfo exists $w.swatchpicker]} {
            set f [$w.swatchpicker childsite]

            if {$col == 4} {
                incr row
                frame $f.swatchFrame$row
                pack $f.swatchFrame$row -side top -anchor nw
                set col 0
            }

            set r [lindex [file split $file] end]
            set num [lindex [split $r "."] 0]
            image create photo img-$r -format "ppm" -file $file
            #Load in the image to diplay on the button and do that.
            
            button $f.swatchFrame$row.swatch$num -image img-$r \
		-command "set $this-filename [netedit getenv HOME]/SCIRun/$num.ppm.cmap2; $this swatch_load $num"
            grid configure $f.swatchFrame$row.swatch$num -row $row \
		-col $col -sticky "nw"
            
            incr col
        }
        set $this-row $row
        set $this-col $col
    }

    method swatch_delete {} {
        # The global $this-filename should be loaded with a xff now. 
	# We want to recover the base
        # filename, essentially the timestamp.  
	# Then we can safely delete it.  This assumes the natural behavior
        # that a swatch will be loaded and immediately deleted.
        global deleteSwatch 
        global $this-row
        global $this-col
         
        # Note:  The following lines MAY break in windows due to 
	# path declaration differences (/ vs. \)
        set basename [file split $deleteSwatch] 
        set basename [lindex $basename end]
        set basename [lindex [split $basename "."] 0]
        set path "[netedit getenv HOME]/SCIRun"
        file delete "$path/$basename.ppm"
        file delete "$path/$basename.ppm.cmap2"

        set w .ui[modname]
        if {[winfo exists $w.swatchpicker]} {
            set f [$w.swatchpicker childsite]
            foreach element [winfo children $f] {
              destroy $element
            }
            set $this-row 0
            set $this-col 0
            create_swatches
        }
    }

    method swatch_load {swatchNum} {
        #  Note:  This is the method in which we must 
	# prepare the global filename for swatch_delete to work on.  
        global deleteSwatch
        set path "[netedit getenv HOME]/SCIRun"
        set deleteSwatch "$path/$swatchNum.ppm.cmap2"
        $this-c load
    }

    method swatch_save {} {
        global $this-filename
        set curdir [pwd]
        set path "[netedit getenv HOME]/SCIRun"
	if { ! [file isdirectory $path] } {
	    file mkdir $path
	}
	
        set numPPM 0
       
        cd $path
        set files [glob -nocomplain "*.ppm"]
        cd $curdir

        set numPPM [clock format [clock seconds] -format {%Y%m%d_%H%M%S}]
        
        set $this-filename "$path/$numPPM.ppm"
        $this-c "saveppm"

	# the "saveppm" command does nothing if the TF has not been changed
	# since it was last saved, which
	# leads to a non-existant $path/$numPPM.ppm
	if { [file exists $path/$numPPM.ppm] } {
	    update_swatches $path/$numPPM.ppm
	}
    }

    method label_widget_columns { frame } {
        frame $frame
        label $frame.name -text "Name" -width 14 -relief groove
        label $frame.color -text "Color" -width 4 -relief groove
        label $frame.shade -text "Solid" -width 8 -relief groove
        label $frame.onoff -text "On" -width 9 -relief groove
        label $frame.empty -text "" -width 3
        pack $frame.name $frame.color $frame.shade \
	    $frame.onoff $frame.empty -side left 
	return $frame
    }


    method ui {} {
        global $this-filename
        global $this-num-entries
        trace vdelete $this-num-entries w "$this unpickle"

        set w .ui[modname]
        if {[winfo exists $w]} {
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

        iwidgets::scrolledframe $w.widgets -hscrollmode none \
	    -vscrollmode static
        iwidgets::scrolledframe $w.swatchpicker -hscrollmode none \
	    -vscrollmode dynamic


        frame $w.controls
        button $w.controls.addtriangle -text "Add Triangle" \
            -command "$this-c addtriangle" -width 11
        button $w.controls.addrectangle -text "Add Rectangle" \
            -command "$this-c addrectangle" -width 11
        button $w.controls.delete -text "Delete" \
            -command "$this-c deletewidget" -width 11
        button $w.controls.undo -text "Undo" \
            -command "$this-c undowidget" -width 11
        button $w.controls.load -text "Load" \
            -command "$this file_load" -width 11
        button $w.controls.save -text "Save" \
            -command "$this file_save" -width 11
        pack $w.controls.addtriangle $w.controls.addrectangle \
	    $w.controls.delete $w.controls.undo \
	    $w.controls.load $w.controls.save \
            -padx 8 -pady 4 -fill x -expand yes -side left
	
	if 1 {
	    frame $w.swatchcontrol
	    button $w.swatchcontrol.saveButton -text "QuickSave" \
		-command "$this swatch_save" -width 20
	    button $w.swatchcontrol.delButton -text "Delete Swatch" \
		-command "$this swatch_delete" -width 20
	    pack $w.swatchcontrol.saveButton $w.swatchcontrol.delButton
	}
	
	
        pack [label_widget_columns $w.title]  -fill x -padx 2 -pady 2
        pack $w.widgets -side top -fill both -expand yes -padx 2
        pack $w.controls -fill x 
        
	add_frame [$w.widgets childsite]
        create_entries
	if 1 {
	    pack $w.swatchcontrol -side top -fill x
	    pack $w.swatchpicker -side top -fill both -expand yes -padx 2
	    create_swatches
	}

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

