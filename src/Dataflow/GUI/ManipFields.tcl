

itcl_class SCIRun_Fields_ManipFields {
    inherit Module
    constructor {config} {
        set name ManipFields
        set_defaults
    }

    method set_defaults {} {
	global $this-manips
	global $this-curmanip
	set $this-manips {Custom}
	set $this-curmanip [lindex [set $this-manips] 0]
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w
	
	iwidgets::tabnotebook $w.tabs -width 300 -height 200
	pack $w.tabs

	$this-c getManips

	addManip Custom

	foreach manip "[set $this-manips]" {
	    addManip $manip
	}

	$w.tabs select 0

	button $w.exec -text execute -command "$this execute"
	pack $w.exec -pady 5
    }

    method execute {} {
	if {"[set $this-curmanip]"!="Custom"} {
	    $this-c needexecute
	}
    }

    method addManip { new_manip } {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    global page[modname]$new_manip
	    global auto_index
	    set page[modname]$new_manip [$w.tabs add -label "$new_manip" \
		    -command "set $this-curmanip $new_manip"]
	    if {"$new_manip"=="Custom"} {
		fm_ui_$new_manip [set page[modname]$new_manip] [modname]
	    } else {
		set available [array names auto_index fm_ui_$new_manip]
		if {"$available"!=""} {
		    fm_ui_$new_manip [set page[modname]$new_manip] [modname]
		} else {
		    def_ui $new_manip [modname]
		}
	    }
	}
    }

    method getCurManip {} {
	return [set $this-curmanip]
    }
}

proc def_ui {new_manip modname} {
    global page$modname$new_manip
    set p [set page$modname$new_manip]
    
    label $p.l -text "Couldn't find the GUI for $new_manip." \
	  -fore darkred -back darkgrey
    pack $p.l -padx 15 -pady 15
}

proc fm_ui_Custom { p modname} {
    frame $p.l0
    frame $p.l01
    frame $p.l1
    frame $p.l2
    frame $p.l3
    frame $p.l4
    pack $p.l0 $p.l01 $p.l1 $p.l2 $p.l3 -side top -fill both -pady 2 -padx 2
    pack $p.l4 -side top -pady 2 -padx 2

    label $p.l0.l -text "SCIRun path: " -width 15 -anch e
    entry $p.l0.path -width 30
    label $p.l01.l -text "Build path: " -width 15 -anch e
    entry $p.l01.build -width 30
    label $p.l1.l -text "Name: " -width 15 -anch e
    entry $p.l1.manip_name -width 30
    label $p.l2.l -text "CFLAGS: " -width 15 -anch e
    entry $p.l2.cflags -width 30
    label $p.l3.l -text "LIBS: " -width 15 -anch e
    entry $p.l3.libs -width 30
    pack $p.l0.l $p.l0.path \
	 $p.l01.l $p.l01.build \
         $p.l1.l $p.l1.manip_name \
	 $p.l2.l $p.l2.cflags \
	 $p.l3.l $p.l3.libs -side left -fill x 

    button $p.l4.edit -text "Edit" -width 10 \
	   -command "edit_cc_file $p $modname"
    button $p.l4.build -text "Build" -width 10 \
	   -command "build_lib $p $modname"
    pack $p.l4.edit $p.l4.build -side left -pady 7 -padx 7
}

proc edit_cc_file { p modname } {
    set path [$p.l0.path get]
    set name [$p.l1.manip_name get]
    set fullpath $path/src/Dataflow/Modules/ManipFields/$name.cc
    puts "fullpath = $fullpath"
    if {"$fullpath"=="" || [file isdirectory $fullpath]} {
	puts "the given path and/or name is invalid.  Please try again."
	return 
    }

    if {[file exists $fullpath]} {
	$modname-c edit $fullpath
    } else {
	$modname-c generate $path $name
    }
}

proc build_lib { p modname } {
    $modname-c rebuildManips [$p.l0.path get] [$p.l01.build get]
}
