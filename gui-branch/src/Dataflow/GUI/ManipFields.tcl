#
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


itcl_class SCIRun_Fields_ManipFields {
    inherit Module
    constructor {config} {
	global editor
	set editor emacs
	set name "ManipFields"
	set_defaults
    }

    method set_defaults {} {
	global $this-manipulationName
	global $this-name-list
	set $this-manipulationName ""
	set $this-mlibs {}
	set $this-mlibpath {}
	set $this-minc {}
	set $this-name-list {}
    }
    # update the libs information 
    method libDone {arg} {
	$this-c libs $arg
	$this reset_cur_libs $arg mlibs
    }

    # update the libpath information 
    method libPathDone {arg} {
	$this-c libpath $arg
	$this reset_cur_libs $arg mlibpath
    }

    # update the include information 
    method includeDone {arg} {
	$this-c inc $arg
	$this reset_cur_libs $arg minc
    }

    method raiseELB {elb callback title itemlist} {
	set window .ui[modname]
	if {[winfo exists $window.$elb]} {
	    destroy $window.$elb
	} 
	toplevel $window.$elb
	makeEditableListBox $window.$elb $callback $title \
		[set $this-$itemlist]
    }

     method reset_cur_libs {arg list} {
	 global $this-$list
	 puts stdout "RETSETTIGN THE LIBS"
	 if {$arg == "clear"} {
	     set $this-$list {}
	 } else {
	     lappend $this-$list $arg
	 }
    }

    method set_cur_libs {list} {
	global $this-mlibs
	set $this-mlibs $list
    }

    method set_cur_libpath {list} {
	global $this-mlibpath
	set $this-mlibpath $list
    }

    method set_cur_inc {list} {
	global $this-minc
	set $this-minc $list
    }

    method set_name {name} {
	set $this-manipulationName [eval $name]
	$this-c loadmanip [eval $name]
    }

    method set_names {namelist} {
	global $this-name-list
	set $this-name-list $namelist
    }
    
    method addManipSettings {f} {
	global $this-mlibs
	global $this-name-list

	# Editable Dropdown Combobox
	iwidgets::combobox $f.name -labeltext "Manipulation:" \
	     -selectioncommand "$this set_name \"$f.name getcurselection\""

	foreach item [set $this-name-list] {
	    puts stdout $item
	    $f.name insert list 0 $item
	}

	frame $f.make -borderwidth 2
	#set the libs needed to link against
	#set libList {Core_Thread Core_Malloc Core_Exceptions}
	set nlib "$this raiseELB libs \"$this libDone\" LIBS mlibs"
	button $f.make.libs -text libs -command $nlib
	
	#set the libpath needed to find the libs
	set lpList {\\$(SCIRUN_SRCTOP)/lib}
	set nlibpath \
	  "$this raiseELB libpath \"$this libPathDone\" LIBPATH mlibpath"
	button $f.make.libpath -text libpath -command $nlibpath
	
	#set the include path to find code we want to link in
	set incList {\\$(SRCTOP)}
	set ninc \
	  "$this raiseELB includes \"$this includeDone\" INCLUDES minc"
	button $f.make.includes -text includes -command $ninc

	pack $f.make.libs $f.make.libpath $f.make.includes \
		-padx 2 -pady 2 -side left
	pack $f.name $f.make -padx 2 -pady 2 -side top
    }
    
    method launchEditor {} {
	global editor
	$this-c launch-editor $editor
    }

    method launchCompile {} {
	$this-c compile
    }
    
    method addCompilation {f} {
	global editor

	#edit the code
	button $f.edit -text edit -command "$this launchEditor"
	entry $f.editorEntry -width 20 \
	    -relief sunken -bd 2 -textvariable editor
	pack $f.edit $f.editorEntry -padx 2 -pady 2 -side left -fill x
    }

    method ui {} {
	global $this-manipulationName
	set window .ui[modname]
	if {[winfo exists $window]} {
	    raise $window
	    return;
	}
	#build up the frames to hold things
	toplevel $window
	frame $window.all -relief groove -borderwidth 2
	label $window.all.frameTitle -text "Manipulate Field"

	#add frame for manipulation specific details
	frame $window.all.options -borderwidth 2
	
	#add frame for compilation related ui
	frame $window.all.compilation -borderwidth 2
	
	#add bottom frame for execute and dismiss buttons
	frame $window.all.control -borderwidth 2

	pack $window.all.frameTitle $window.all.options \
		$window.all.compilation $window.all.control \
		-side top


	#add the various settings ui
	addManipSettings $window.all.options
	addCompilation $window.all.compilation
	
	# add the go away parts
	set n "$this-c needexecute"	
	button $window.all.control.execute -text Execute -command $n
		#compile the code
	button $window.all.control.compile -text compile \
		-command "$this launchCompile"
	button $window.all.control.dismiss -text Dismiss \
		-command "destroy $window"
	pack $window.all.control.execute $window.all.control.compile \
		$window.all.control.dismiss -padx 2 -pady 2 -side left
	pack $window.all
    }
}



#editable list box

global newitem
global selindex
set newitem "add path here"
set selindex -1

proc cacheSelectedIndex {slb} {
    global selindex
    set selindex [$slb curselection]
}

proc addItem {slb} {
    global newitem
    $slb insert end $newitem 
}

proc delItem {slb} {
    global selindex
    if {$selindex >= 0} {
	$slb delete $selindex
	set selindex -1
    }
}

proc sendString {cb slb} {
    set s [$slb size]
    puts stdout $s
    eval $cb clear
    for {set i 0} {$i < $s} {incr i 1} {
	eval $cb \"[$slb get $i]\"
    }
}

proc makeEditableListBox {w callback label items} {

    frame $w.elb
    frame $w.elb.top
    frame $w.elb.bottom
    frame $w.elb.fini

    iwidgets::scrolledlistbox $w.elb.top.slb -vscrollmode static \
	    -hscrollmode dynamic \
	    -selectioncommand "cacheSelectedIndex $w.elb.top.slb" \
	    -scrollmargin 3 -labelpos n -labeltext $label
    

    button $w.elb.top.minus -text "-" -command "delItem $w.elb.top.slb"
    pack $w.elb.top.slb $w.elb.top.minus -padx 3 -side left
    foreach item $items {
	$w.elb.top.slb insert end $item
	puts stdout $item
    }

    label $w.elb.bottom.plusLabel -text add
    entry $w.elb.bottom.plusEntry -width 30 \
	    -relief sunken -bd 2 -textvariable newitem
    button $w.elb.bottom.plus -text "+" -command "addItem $w.elb.top.slb"
    pack $w.elb.bottom.plusLabel $w.elb.bottom.plusEntry \
	    $w.elb.bottom.plus -padx 3 -side left

    button $w.elb.fini.ok -text OK \
	    -command "sendString \"$callback\" $w.elb.top.slb"
    button $w.elb.fini.cancel -text Cancel -command "destroy $w"
    pack $w.elb.fini.ok $w.elb.fini.cancel -side left 
    pack $w.elb.top $w.elb.bottom $w.elb.fini -side top
    pack $w.elb
}







