itcl_class Uintah_DataIO_ArchiveReader { 

    inherit Module 
    
    protected filedir
    protected filename

    constructor {config} { 
        set name ArchiveReader 
        set_defaults
    } 
  
    method filedir { filebase } {
	set n [string last "/" "$filebase"]
	if { $n != -1} {
	    return [ string range $filebase 0 $n ]
	} else {
	    return ""
	}
    }

    method filename { filebase } {
	set n [string last "/" "$filebase"]
	if { $n != -1} {
	    return [ string range $filebase [eval $n + 1] end]
	} else {
	    return ""
	}
    }
	
    method set_defaults {} { 
	global $this-filebase 
	global $this-oldfilebase
	set $this-filebase ""
	set $this-oldfilebase ""
    } 
  

    method ui {} { 
        set w .ui[modname] 
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	global $this-startFrame
	global $this-endFrame
	global $this-animate
	global $this-increment
	global $this-tcl_status
	global $this-filebase
	global env

        toplevel $w 
        wm minsize $w 100 50 
  
        set n "$this-c needexecute " 
  
        frame $w.f1 -relief groove -borderwidth 2
	pack $w.f1 -in $w -side left
	
	
	if { [string compare [set $this-filebase] ""] == 0 } {
	    if { [info exists env(PSE_DATA)] } {
		set $this-filebase $env(PSE_DATA)
	    } else {
		set $this-filebase $env(PWD)
 	    }
	    $this makeFilebox $w.f1
	} else {
	    $this makeFilebox $w.f1
	}
	
	button $w.f1.select -text Select -command "$this selectfile"
	pack $w.f1.select -side left -padx 2 -pady 2
	button $w.f1.close -text Close -command "wm withdraw $w"
	pack $w.f1.close -side left -padx 2 -pady 2

    } 

    method selectfile {} {
	set w .ui[modname]
	$this-c needexecute
    }

    method makeFilebox { w } {

	set $this-oldfilebase [set $this-filebase]
	frame $w.f

	frame $w.f.bro
	
	frame $w.f.bro.dir
	label $w.f.bro.dir.dirsl -text "Directories/Archives"
	listbox $w.f.bro.dir.dirs -relief sunken \
	    -yscrollcommand "$w.f.bro.dir.dirss1 set" \
	    -xscrollcommand "$w.f.bro.dir.dirss2 set"
	set dirs $w.f.bro.dir.dirs

	bind $w.f.bro.dir.dirs <Double-Button-1> "$this fbdirs %y $w $dirs"
	scrollbar $w.f.bro.dir.dirss1 -relief sunken \
	    -command "$w.f.bro.dir.dirs yview"
	scrollbar $w.f.bro.dir.dirss2 -relief sunken -orient horizontal \
	    -command "$w.f.bro.dir.dirs xview"
	pack $w.f.bro.dir.dirsl -in $w.f.bro.dir -side top -padx 2 -pady 2 \
	    -anchor w
	pack $w.f.bro.dir.dirss2 -in $w.f.bro.dir -side bottom -padx 2 \
	    -pady 2 -anchor s -fill x
	pack $w.f.bro.dir.dirs -in $w.f.bro.dir -side left -padx 2 -pady 2 \
	    -anchor w -expand yes -fill x
	pack $w.f.bro.dir.dirss1 -in $w.f.bro.dir -side right -padx 2 \
	    -pady 2 -anchor e -fill y

	pack $w.f.bro.dir -in $w.f.bro -side left -padx 2 \
	    -pady 2 -expand 1 -fill x

	frame $w.f.sel
	label $w.f.sel.sell -text Selection
	entry $w.f.sel.sel -relief sunken -width 40 \
	    -textvariable $this-filebase
	bind $w.f.sel.sel <Return> "$this fbsel $w $dirs "
	pack $w.f.sel.sell -in $w.f.sel -side top -padx 2 -pady 2 -anchor w
	pack $w.f.sel.sel -in $w.f.sel -side bottom -padx 2 -pady 2 \
	     -anchor w -fill x

	pack $w.f.bro $w.f.sel -in $w.f -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
	pack $w.f


	$this fbupdate $w $dirs	
	
    }

    method fbsel {w dirs } {
    if [file isdirectory [set $this-filebase]] {
	fbcd $w [set $this-filebase] $dirs 
#    } else {
#	eval $this selectfile
#    }
    }

    method fbupdate {w dirs} {
	$dirs delete 0 end
	foreach i [lsort [glob -nocomplain [set $this-filebase]/.* [set $this-filebase]/*]] {
	    if [file isdirectory $i] {
		$dirs insert end [file tail $i]
	    }
	}
	update
    }
    method fbdirs {y w dirs } {


	set ind [$dirs nearest $y]
	$dirs selection set $ind
	set dir [$dirs get $ind]

	if [expr [string compare "." $dir] == 0] {
	    return
	} elseif [expr [string compare ".." $dir] == 0] {
	    $this fbcd $w [file dirname [set $this-filebase]] $dirs 
	} else {
	    $this fbcd $w [set $this-filebase]/$dir $dirs 
	}
    }

    method fbpath {w dirs } {

	$this fbcd $w [set $this-filebase] $dirs 
    }

    method fbcd {w dir dirs } {

	if [file isdirectory $dir] {
	    set $this-filebase $dir
	    set $this-oldfilebase [set $this-filebase]
	    $this fbupdate $w $dirs 
	} else {
	    set $this-filebase [set $this-oldfilebase]
	}
    }

}
