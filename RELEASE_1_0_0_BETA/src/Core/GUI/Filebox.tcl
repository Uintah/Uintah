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

#  Filebox.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Mar. 1994
#  Copyright (C) 1994 SCI Group

proc updateFilebox { w } {
    set dirs $w.f.bro.dir.dirs
    set files $w.f.bro.file.files
    fbupdate $w $dirs $files
}

proc setFileboxToHomeDir { w } {

    global env

    set dirs $w.f.bro.dir.dirs
    set files $w.f.bro.file.files

    fbcd $w $env(PWD) $dirs $files
}

proc makeFileboxWithFilter { w var command cancel filter } {

    global $w-filter $w-path $w-oldpath $w-oldsel 
    global $var


    set $w-filter "$filter"

    set $w-oldsel ""

    global env
    if { [info exists env(PSE_DATA)] } {
	set $w-path $env(PSE_DATA)
    } else {
	set $w-path $env(PWD)
    }
    set $w-oldpath [set $w-path]

    frame $w.f

    frame $w.f.bro

    frame $w.f.bro.file
    label $w.f.bro.file.filesl -text Files
    listbox $w.f.bro.file.files -relief sunken \
	    -yscrollcommand "$w.f.bro.file.filess1 set" \
	    -xscrollcommand "$w.f.bro.file.filess2 set"
    set files $w.f.bro.file.files
#   tk_listboxSingleSelect $files
    bind $w.f.bro.file.files <Button-1> \
	    "fbselect %y $w $files $var"
    bind $w.f.bro.file.files <Double-Button-1> \
	    "fbchoose %y $w $files $var \"$command\""
    scrollbar $w.f.bro.file.filess1 -relief sunken \
	    -command "$w.f.bro.file.files yview"
    scrollbar $w.f.bro.file.filess2 -relief sunken -orient horizontal \
	    -command "$w.f.bro.file.files xview"
    pack $w.f.bro.file.filesl -in $w.f.bro.file -side top -padx 2 \
	    -pady 2 -anchor w
    pack $w.f.bro.file.filess2 -in $w.f.bro.file -side bottom -padx 2 \
	    -pady 2 -anchor s -fill x
    pack $w.f.bro.file.files -in $w.f.bro.file -side left -padx 2 \
	    -pady 2 -anchor w
    pack $w.f.bro.file.filess1 -in $w.f.bro.file -side right -padx 2 \
	    -pady 2 -anchor e -fill y

    frame $w.f.bro.dir
    label $w.f.bro.dir.dirsl -text Directories
    listbox $w.f.bro.dir.dirs -relief sunken \
	    -yscrollcommand "$w.f.bro.dir.dirss1 set" \
	    -xscrollcommand "$w.f.bro.dir.dirss2 set"
    set dirs $w.f.bro.dir.dirs
#    tk_listboxSingleSelect $dirs
    bind $w.f.bro.dir.dirs <Double-Button-1> "fbdirs %y $w $dirs $files"
    scrollbar $w.f.bro.dir.dirss1 -relief sunken \
	    -command "$w.f.bro.dir.dirs yview"
    scrollbar $w.f.bro.dir.dirss2 -relief sunken -orient horizontal \
	    -command "$w.f.bro.dir.dirs xview"
    pack $w.f.bro.dir.dirsl -in $w.f.bro.dir -side top -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.bro.dir.dirss2 -in $w.f.bro.dir -side bottom -padx 2 \
	    -pady 2 -anchor s -fill x
    pack $w.f.bro.dir.dirs -in $w.f.bro.dir -side left -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.bro.dir.dirss1 -in $w.f.bro.dir -side right -padx 2 \
	    -pady 2 -anchor e -fill y

    pack $w.f.bro.dir $w.f.bro.file -in $w.f.bro -side left -padx 2 \
	    -pady 2 -expand 1 -fill x

    frame $w.f.filt
    label $w.f.filt.filtl -text Filter
    entry $w.f.filt.filt -relief sunken -width 40 -textvariable $w-filter
    bind $w.f.filt.filt <Return> "fbupdate $w $dirs $files"
    pack $w.f.filt.filtl -in $w.f.filt -side top -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.filt.filt -in $w.f.filt -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x

    frame $w.f.path
    label $w.f.path.pathl -text Path
    entry $w.f.path.path -relief sunken -width 40 -textvariable $w-path
    bind $w.f.path.path <Return> "fbpath $w $dirs $files"
    pack $w.f.path.pathl -in $w.f.path -side top -padx 2 -pady 2 -anchor w
    pack $w.f.path.path -in $w.f.path -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x

    frame $w.f.sel
    label $w.f.sel.sell -text Selection
    entry $w.f.sel.sel -relief sunken -width 40 -textvariable $var
    bind $w.f.sel.sel <Return> "fbsel $w $dirs $files $var \"$command\""
    pack $w.f.sel.sell -in $w.f.sel -side top -padx 2 -pady 2 -anchor w
    pack $w.f.sel.sel -in $w.f.sel -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x


    frame $w.f.extra

    frame $w.f.but
    button $w.f.but.ok -text OK -command $command
    button $w.f.but.filt -text Filter -command "fbupdate $w $dirs $files"
    button $w.f.but.home -text Home -command "fbcd $w $env(PWD) $dirs $files"
    button $w.f.but.data -text Data -command "fbcd $w $w-path $dirs $files"
    button $w.f.but.cancel -text Close -command $cancel
    pack $w.f.but.ok -in $w.f.but -side left -padx 2 -pady 2 -anchor w
    pack $w.f.but.cancel -in $w.f.but -side right -padx 2 -pady 2 \
	    -anchor e
    pack $w.f.but.filt -in $w.f.but -side left -padx 2 -pady 8 -expand 1
    pack $w.f.but.home -in $w.f.but -side left -padx 2 -pady 8 -expand 1
    pack $w.f.but.data -in $w.f.but -side left -padx 2 -pady 8 -expand 1

    pack $w.f.filt $w.f.path $w.f.bro $w.f.sel $w.f.extra \
	    $w.f.but -in $w.f -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
    pack $w.f

    fbupdate $w $dirs $files
}

proc makeFilebox {w var command cancel} {
    makeFileboxWithFilter $w $var $command $cancel "*"
}

proc fbsel {w dirs files var command} {
    global $w-path $w-oldpath $w-oldsel $var

    if [file isdirectory [set $var]] {
	fbcd $w [set $var] $dirs $files
    } else {
	eval $command
    }
}

proc fbdirs {y w dirs files} {
    global $w-path

    set ind [$dirs nearest $y]
    $dirs selection set $ind
    set dir [$dirs get $ind]

    if [expr [string compare "." $dir] == 0] {
	return
    } elseif [expr [string compare ".." $dir] == 0] {
	fbcd $w [file dirname [set $w-path]] $dirs $files
    } else {
	fbcd $w [set $w-path]/$dir $dirs $files
    }
}

proc fbpath {w dirs files} {
    global $w-path

    fbcd $w [set $w-path] $dirs $files
}

proc fbcd {w dir dirs files} {
    global $w-path $w-oldpath

    if [file isdirectory $dir] {
	set $w-path $dir
	set $w-oldpath [set $w-path]
	fbupdate $w $dirs $files
    } else {
	set $w-path [set $w-oldpath]
    }
}

proc fbupdate {w dirs files} {
    global $w-filter $w-path

    $dirs delete 0 end
    foreach i [lsort [glob -nocomplain [set $w-path]/.* [set $w-path]/*]] {
	if [file isdirectory $i] {
	    $dirs insert end [file tail $i]
	}
    }
    $files delete 0 end
    foreach i [lsort [glob -nocomplain [set $w-path]/{[set $w-filter]}]] {
	if [file isfile $i] {
	    $files insert end [file tail $i]
	}
    }

    update
}    

proc fbchoose {y w files var command} {
    fbselect $y $w $files $var
    global $var
    eval $command
}

proc fbselect {y w files var} {
    global $w-path $w-oldsel $var

    set ind [$files nearest $y]
    $files selection set $ind
    set $var [set $w-path]/[$files get $ind]
    set $w-oldsel [set $var]
}
