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


catch {rename EditTransferFunc2 ""}

itcl_class SCIRun_Visualization_EditTransferFunc2 {
    inherit Module
    constructor {config} {
	set name EditTransferFunc2
	set_defaults
    }
    
    method set_defaults {} {
	global $this-faux
	set $this-faux 0
	global $this-histo
	set $this-histo 0.5

	global $this-num-entries
	set $this-num-entries 2
	
	global $this-marker
	trace variable $this-marker w "$this unpickle"

        global $this-filename
        set $this-filename "MyTransferFunction"
	
        global $this-filetype
        set $this-filetype Binary
    }
    
    method unpickle {a b c} {
	global $this-marker
	$this-c unpickle
	trace vdelete $this-marker w "$this unpickle"
    }
    
    method raise_color {col color colMsg} {
	global $color
	set window .ui[modname]
	if {[winfo exists $window.color]} {
	    SciRaise $window.color
	    return
	} else {
	    # makeColorPicker now creates the $window.color toplevel.
	    makeColorPicker $window.color $color \
		"$this set_color $col $color $colMsg" \
		"destroy $window.color"
	}
    }
    
    method set_color {col color colMsg} {
	global $color
	global $color-r
	global $color-g
	global $color-b
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]

	set window .ui[modname]
	$col config -background [format #%04x%04x%04x $ir $ig $ib]  \
		    -activebackground [format #%04x%04x%04x $ir $ig $ib]
					
	 puts "$ir $ig $ib"
				      
	 $this-c $colMsg
    }

    method create_entries {} {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set widgets [$w.widgets childsite]

	    # Create the new variables and entries if needed.
	    for {set i 0} {$i < [set $this-num-entries]} {incr i} {
		
		if { [catch { set t [set $this-names-$i] } ] } {
		    set $this-name-$i default-$i
		}
		if { [catch { set t [set $this-$i-color-r]}] } {
		    set $this-$i-color-r 0.5
		}
		if { [catch { set t [set $this-$i-color-g]}] } {
		    set $this-$i-color-g 0.5
		}
		if { [catch { set t [set $this-$i-color-b]}] } {
		    set $this-$i-color-b 1.0
		}
		if { [catch { set t [set $this-$i-color-a]}] } {
		    set $this-$i-color-a 0.7
		}

		if {![winfo exists $widgets.e-$i]} {
		    frame $widgets.e-$i
		    entry $widgets.e-$i.name \
			-textvariable $this-name-$i -width 16
		    set cmmd "$this raise_color $widgets.e-$i.color $this-$i-color color_change-$i"
		    button $widgets.e-$i.color -width 8 \
			-command $cmmd
                    checkbutton $widgets.e-$i.shade -text "" -padx 26 -justify center \
                        -relief flat -variable $this-shadeType-$i -onvalue 1 -offvalue 0 \
                        -anchor w -command "$this-c shadewidget-$i"
                    checkbutton $widgets.e-$i.on -text "" -padx 6 -justify center \
                        -relief flat -variable $this-on-$i -onvalue 1 -offvalue 0 \
                        -anchor w -command "$this-c toggleon-$i"
                    $widgets.e-$i.on select;   # set it to on.
		    pack $widgets.e-$i.name $widgets.e-$i.color $widgets.e-$i.shade \
			$widgets.e-$i.on -side left
		    pack $widgets.e-$i 
		}

		set ir [expr int([set $this-$i-color-r] * 65535)]
		set ig [expr int([set $this-$i-color-g] * 65535)]
		set ib [expr int([set $this-$i-color-b] * 65535)]
                $widgets.e-$i.color configure \
			-background [format #%04x%04x%04x $ir $ig $ib] \
                    	-activebackground [format #%04x%04x%04x $ir $ig $ib]
	    }

	    # Destroy all the left over entries from prior runs.
	    while {[winfo exists $widgets.e-$i]} {
		destroy $widgets.e-$i
		incr i
	    }
	}
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
        set defext ".xff"

        # name to appear initially
        set defname "TransferFunc01"
        set title "Save Transfer Function"

        # file types to appers in filter box
        set types {
            {{All Files}       {.xff}   }
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
        set defext ".xff"
        set title "Open SCIRun Transfer Function file"

        # file types to appers in filter box
        set types {
            {{SCIRun Transfer Function}     {.xff}      }
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

    method ui {} {
        global $this-filename
	global $this-num-entries
	trace vdelete $this-num-entries w "$this unpickle"

	set w .ui[modname]
	if {[winfo exists $w]} {
	    create_gl
	    return
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
        create_gl

	iwidgets::scrolledframe $w.widgets -hscrollmode none -vscrollmode dynamic

	frame $w.title
	label $w.title.name -text "Widget Name" \
	    -width 16 -relief groove
	label $w.title.color -text "Color" -width 8 -relief groove
	label $w.title.shade -text "Flat Shaded" -width 10 -relief groove
        label $w.title.onoff -text "On" -width 4 -relief groove
	label $w.title.empty -text "" -width 3
	pack $w.title.name $w.title.color $w.title.shade $w.title.onoff $w.title.empty \
	    -side left 

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
	    $w.controls.delete $w.controls.undo $w.controls.load $w.controls.save \
	    -padx 8 -pady 4 -fill x -expand yes -side left

	pack $w.title  -fill x -padx 2 -pady 2
	pack $w.widgets -side top -fill both -expand yes -padx 2
	pack $w.controls -fill x 

	create_entries

	makeSciButtonPanel $w $w $this "\"Reset\" \"$this-c reset_gui\" \"\""
	moveToCursor $w
    }
	    
    method bind_events {w} {
	# every time the OpenGL widget is displayed, redraw it
	bind $w <Expose> "$this-c expose"
	bind $w <ButtonPress-1> "$this-c mouse push %x %y %b 0"
	bind $w <ButtonPress-2> "$this-c mouse push %x %y %b 0"
	bind $w <ButtonPress-3> "$this-c mouse push %x %y %b 1"
	bind $w <Button1-Motion> "$this-c mouse motion %x %y"
	bind $w <Button2-Motion> "$this-c mouse motion %x %y"
	bind $w <Button3-Motion> "$this-c mouse motion %x %y"
	bind $w <ButtonRelease-1> "$this-c mouse release %x %y %b"
	bind $w <ButtonRelease-2> "$this-c mouse release %x %y %b"
	bind $w <ButtonRelease-3> "$this-c mouse release %x %y %b"

	# controls for pan and zoom and reset
	bind $w <Shift-ButtonPress-1> "$this-c mouse translate start %x %y"
	bind $w <Shift-Button1-Motion> "$this-c mouse translate move %x %y"
	bind $w <Shift-ButtonRelease-1> "$this-c mouse translate end %x %y"
	bind $w <Shift-ButtonPress-2> "$this-c mouse reset %x %y 1"
	bind $w <Shift-Button2-Motion> "$this-c mouse reset %x %y 1"
	bind $w <Shift-ButtonRelease-2> "$this-c mouse reset %x %y 1"
	bind $w <Shift-ButtonPress-3> "$this-c mouse scale start %x %y"
	bind $w <Shift-Button3-Motion> "$this-c mouse scale move %x %y"
	bind $w <Shift-ButtonRelease-3> "$this-c mouse scale end %x %y"

	bind $w <Destroy> "$this-c closewindow"		
    }
	    
    method create_gl {} {
        set w .ui[modname]
        if {[winfo exists $w.f.gl]} {
            raise $w
        } else {
            set n "$this-c needexecute"
	    
            frame $w.f.gl -relief groove -borderwidth 2
            pack $w.f.gl -padx 2 -pady 2
            # create an OpenGL widget
            opengl $w.f.gl.gl -geometry 512x256 -doublebuffer true -direct true \
                -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2
	    bind_events $w.f.gl.gl
            # place the widget on the screen
            pack $w.f.gl.gl -fill both -expand 1
            # histogram opacity
            scale $w.f.gl.shisto -variable $this-histo \
                -from 0.0 -to 1.0 -label "Histogram Opacity" \
                -showvalue true -resolution 0.001 \
                -orient horizontal -command "$this-c redraw true"
            pack $w.f.gl.shisto -side top -fill x -padx 4
            # faux shading
            frame $w.f.f0 -relief groove -borderwidth 2
            pack $w.f.f0 -padx 2 -pady 2 -fill x
            checkbutton $w.f.f0.faux -text "Opacity Modulation (Faux Shading)" \
                -relief flat -variable $this-faux -onvalue 1 -offvalue 0 \
                -anchor w -command "$n; $this-c redrawcmap"
            pack $w.f.f0.faux -side top -fill x -padx 4
        }
    }
}
