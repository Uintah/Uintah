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

# GUI for DataIO_Readers_MDSPlusFieldReader module
# by Allen R. Sanderson
# March 2002

itcl_class DataIO_Readers_MDSPlusFieldReader {
    inherit Module
    constructor {config} {
        set name MDSPlusFieldReader
        set_defaults
    }

    method set_defaults {} {
	global $this-serverName
	global $this-treeName
	global $this-shotNumber
	global $this-sliceNumber
	global $this-sliceRange
	global $this-sliceStart
	global $this-sliceStop
	global $this-sliceSkip

	global $this-bPressure
	global $this-bBField
	global $this-bVField
	global $this-bJField

	set $this-serverName "atlas.gat.com"
	set $this-treeName "NIMROD"
	set $this-shotNumber "10089"
	set $this-sliceNumber "0"
	set $this-sliceRange 0
	set $this-sliceStart 0
	set $this-sliceStop 0
	set $this-sliceSkip 1

	set $this-bPressure 0
	set $this-bBField 0
	set $this-bVField 0
	set $this-bJField 0

	set $this-space 0
	set $this-mode 0
    }

    method ui {} {
	global $this-serverName
	global $this-treeName
	global $this-shotNumber
	global $this-sliceNumber
	global $this-sliceRange
	global $this-sliceStart
	global $this-sliceStop

	global $this-bPressure
	global $this-bBField
	global $this-bVField
	global $this-bJField

	global $this-space
	global $this-mode

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	labelEntry $w.server "Server" $this-serverName
	labelEntry $w.tree   "Tree"   $this-treeName
	labelEntry $w.shot   "Shot"   $this-shotNumber
	labelEntry $w.slice  "Slice"  $this-sliceNumber

	frame $w.range

	checkbutton $w.range.check -text "Range" -variable $this-sliceRange
	pack $w.range.check -side left

	labelEntry $w.range.start "Start" $this-sliceStart
	labelEntry $w.range.stop  "Stop"  $this-sliceStop
	labelEntry $w.range.skip  "Skip"  $this-sliceSkip

	pack $w.range.start $w.range.stop $w.range.skip -side top


	frame $w.space

	radiobutton $w.space.realspace -text "Realspace" -width 10 -anchor w -just left -variable $this-space -value 0

	radiobutton $w.space.perturbed -text "Perturbed" -width 10 -anchor w -just left -variable $this-space -value 1

	pack $w.space.realspace $w.space.perturbed -side left

	frame $w.mode

	label $w.mode.title -text "Mode:"  -width 6 -anchor w -just left

	radiobutton $w.mode.0 -text "0" -width 2 -anchor w -just left -variable $this-mode -value 0
	radiobutton $w.mode.1 -text "1" -width 2 -anchor w -just left -variable $this-mode -value 1
	radiobutton $w.mode.2 -text "2" -width 2 -anchor w -just left -variable $this-mode -value 2
	radiobutton $w.mode.sum -text "Sum" -width 4 -anchor w -just left -variable $this-mode -value 3

	pack $w.mode.title $w.mode.0 $w.mode.1 $w.mode.2 $w.mode.sum -side left

	frame $w.check

	checkbutton $w.check.cb1 -text "Pressure" -variable $this-bPressure
	checkbutton $w.check.cb2 -text "B Field" -variable $this-bBField
	checkbutton $w.check.cb3 -text "V Field" -variable $this-bVField
	checkbutton $w.check.cb4 -text "J Field" -variable $this-bJField

	pack $w.check.cb1 -side left
	pack $w.check.cb2 -side left -padx 25
	pack $w.check.cb3 -side left -padx 25
	pack $w.check.cb4 -side left -padx 25

	pack $w.range $w.space $w.mode $w.check -padx 10 -pady 10

	button $w.button -text "Download" -command "$this-c needexecute"

	pack $w.button -padx 10 -pady 10
    }

    method labelEntry { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5 -pady 5
	label $win.l -text $text1  -width 6 -anchor w -just left
	label $win.c  -text ":" -width 1 -anchor w -just left 
	entry $win.e -text $text2 -width 20 -just left -fore darkred
	pack $win.l $win.c -side left
	pack $win.e -padx 5 -side left
    }
}
