itcl_class Fusion_DataIO_MDSPlusFieldReader {
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

	global $this-bPressure
	global $this-bBField
	global $this-bVField

	set $this-serverName "atlas.gat.com"
	set $this-treeName "NIMROD"
	set $this-shotNumber "10089"
	set $this-sliceNumber "0"

	set $this-bPressure 0
	set $this-bBField 0
	set $this-bVField 0
    }

    method ui {} {
	global $this-serverName
	global $this-treeName
	global $this-shotNumber
	global $this-sliceNumber

	global $this-bPressure
	global $this-bBField
	global $this-bVField

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

	frame $w.check

	checkbutton $w.check.cb1 -text "Pressure" -variable $this-bPressure
	checkbutton $w.check.cb2 -text "B Field" -variable $this-bBField
	checkbutton $w.check.cb3 -text "V Field" -variable $this-bVField

	pack $w.check.cb1 -side left
	pack $w.check.cb2 -side left -padx 25
	pack $w.check.cb3 -side left -padx 25

	pack $w.check -padx 10 -pady 10

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







