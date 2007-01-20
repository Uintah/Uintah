#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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
 #  ChooseModule.tcl: The ChooseModule UI
 #  Written by:
 #   Allen R. Sanderson
 #   SCI Institute
 #   University of Utah
 #   March 2006
 #   Copyright (C) 2006 SCI Group
 ##

itcl_class ChooseModule {
    inherit Module
    protected port_list ""
    protected list_frame ""

    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} {
            return
        }
        build_top_level
    }

    method build_top_level {} {
        set w .ui[modname] 
        
        if {[winfo exists $w]} { 
            return
        } 
	
        toplevel $w 
	wm withdraw $w
	
	wm minsize $w 200 100

	frame $w.main -relief flat
	pack $w.main -fill both -expand yes


	#frame for tabs
	frame $w.main.options
	pack $w.main.options -padx 2 -pady 2 -side top -fill both -expand 1
	#frame for display
	frame $w.main.options.disp -borderwidth 2
	pack $w.main.options.disp -padx 2 -pady 2 -side left \
	    -fill both -expand 1

	# Tabs
	iwidgets::labeledframe $w.main.options.disp.frame_title \
	    -labelpos nw -labeltext "Port Control"
	set dof [$w.main.options.disp.frame_title childsite]

	iwidgets::tabnotebook $dof.tabs -raiseselect true
	pack $dof.tabs -side top -fill both -expand yes

        add_default_tab $dof
        add_select_tab $dof
        toggle_use_first_valid

	$dof.tabs view 0
	$dof.tabs configure -tabpos "n"
	
	pack $w.main.options.disp.frame_title -side top -expand yes -fill both

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    
    method add_default_tab { dof } {
        #-----------------------------------------------------------
        # Standard controls
        #-----------------------------------------------------------
	set tab [$dof.tabs add -label "Basic"]
        
        
	frame $tab.v
	pack $tab.v -side top -e y -f both -padx 5 -pady 5
        

	radiobutton $tab.v.radio \
	    -text "Use first valid port.   Port sent:" \
	    -variable $this-use-first-valid \
	    -value 1 \
	    -command "$this toggle_use_first_valid"

	entry $tab.v.entry -textvariable $this-port-valid-index -state disabled

	pack $tab.v.radio -side left
	pack $tab.v.entry -side left -padx 10

	frame $tab.s
	pack $tab.s -side top -e y -f both -padx 5 -pady 5
	
	radiobutton $tab.s.radio -text "Use use selected port:" \
	    -variable $this-use-first-valid \
	    -value 0 \
	    -command "$this toggle_use_first_valid"

	entry $tab.s.entry -textvariable $this-port-selected-index

	bind $tab.s.entry <Return> "$this-c needexecute"

	pack $tab.s.radio -side left
	pack $tab.s.entry -side left -padx 10


	Tooltip $tab.v "Module will iterate over ports\nand use the first one with a valid handle."


	TooltipMultiline $tab.v.radio \
            "Specify the input port that should be routed to the output port.\n" \
            "Index is 0 based (ie: the first port is index 0, the second port 1, etc.)"
	TooltipMultiline $tab.v.entry \
            "Specify the input port that should be routed to the output port.\n" \
            "Index is 0 based (ie: the first port is index 0, the second port 1, etc.)"

    }
    
    method add_select_tab { dof } {
        set tab [$dof.tabs add -label "Select Port"]
        make_frames $tab

    }

    method toggle_use_first_valid {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
            set dof [$w.main.options.disp.frame_title childsite]
            set tab [$dof.tabs childsite "Basic"]
            if {[winfo exists $tab]} {
                # grey out port stuff if checked
                if {[set $this-use-first-valid]} {
                    $tab.s.entry configure -state disabled
                    $dof.tabs pageconfigure "Select Port" -state disabled
                } else {
                    $tab.s.entry configure -state normal
                    $dof.tabs pageconfigure "Select Port" -state normal
                }
            }
	}
    }

    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }

    method make_frames { tab } {
        frame $tab.f 
        pack $tab.f  -side left -expand yes -fill both
        frame $tab.f.1
        pack $tab.f.1 -side left  -expand yes -fill both
        buildControlFrame $tab.f.1
        set list_frame $tab.f
    }

    method build_frames {} {
        set w .ui[modname] 
        if {![winfo exists $w]} { 
            return
        }
        set dof [$w.main.options.disp.frame_title childsite]
        set tab [$dof.tabs childsite "Select Port"]
        make_frames $tab
    }

    method build_port_list {} {
	set c "$this-c needexecute"

	buildControlFrame $list_frame.1
	for {set i 0} { $i < [llength $port_list] } { incr i } {
	    set port_name [lindex $port_list $i]

	    set lvar [string tolower $port_name]

	    regsub \\. $lvar _ lvar
	    radiobutton $list_frame.1.1.$lvar -text $port_name \
		-variable $this-port-selected-index -command $c -value $i
	    pack $list_frame.1.1.$lvar -side top -anchor w \
                 -expand yes -fill x
#	    if { $port_name == [set $this-port-selected-index] } {
#		$list_frame.1.1.$lvar invoke
#	    }
	}
    }

    method buildControlFrame { name } {
        
	if { [winfo exists $name.1] } {
	    destroy $name.1
	}
        
	frame $name.1 -relief groove -borderwidth 2
	pack $name.1 -side left  -expand yes -fill both -padx 2
    }
    
    method destroy_frames {} {
	set w .ui[modname] 
        
	destroy $list_frame
	set port_list ""
	set list_frame ""
    }

    method set_ports { args } {
	set port_list $args;
    }

}


