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


itcl_class MatlabInterface_DataIO_MatlabMatricesReader {
	inherit Module

	constructor {config} {
		set name MatlabMatricesReader
		set_defaults
	}

	method set_defaults {} {
		global $this-filename
		global $this-matrixinfotexts
		global $this-matrixnames
		global $this-matrixname
		global $this-matriceslistbox		
		global $this-filename-set
		global $this-portsel
		global $this-numport
		
		set $this-filename ""
		set $this-matrixinfotexts ""
		set $this-matrixnames ""
		set $this-matrixname ""
		set $this-matriceslistbox ""
		set $this-filename-set ""
		set $this-portsel 0
		set $this-numport 6
		
	}


	method ui {} {

		global $this-filename
		global $this-matrixinfotexts		
		global $this-matrixnames
		global $this-matrixname
		global $this-matriceslistbox		
		global $this-filename-entry
		global $this-port
		global $this-numport
		global $this-portsel

		set w .ui[modname]

		# test whether gui is already out there
		# raise will move the window to the front
		# so the user can modify the settings

		if {[winfo exists $w]} {
			raise $w
			return
		}

		# create a new gui window

		toplevel $w 

		iwidgets::labeledframe $w.fileframe -labeltext "SELECT MATLAB FILE"
		set childfile [$w.fileframe childsite]
		pack $w.fileframe -fill x

		frame $childfile.f1
		frame $childfile.f2
		
		pack $childfile.f1 $childfile.f2 -side top -fill x -expand yes

		label $childfile.f1.label -text ".MAT FILE "
		entry $childfile.f1.file -textvariable $this-filename
		set $this-filename-entry $childfile.f1.file  
		button $childfile.f2.open -text "Open" -command [format "%s OpenMatfile" $this]
		button $childfile.f2.browse -text "Browse" -command [format "%s ChooseFile" $this]

		pack $childfile.f1.label -side left -padx 3p -pady 2p -padx 4p
		pack $childfile.f1.file -side left -fill x -expand yes -padx 3p -pady 2p
		pack $childfile.f2.browse -side right -padx 3p -pady 2p -anchor e
		pack $childfile.f2.open -side right -padx 3p -pady 2p -anchor e

		iwidgets::labeledframe $w.matrixframe -labeltext "SELECT MATLAB MATRIX"
		set childframe [$w.matrixframe childsite]
		pack $w.matrixframe -fill both -expand yes

		frame $childframe.portframe 
		pack $childframe.portframe -fill x -pady 4p
		set $this-port $childframe.portframe
		
		for {set x 0} {$x < [set $this-numport]} {incr x} {
			button [set $this-port].$x -text [format "Port %d" [expr $x + 1]] -command [format "%s SetPort %d" $this $x]
			pack [set $this-port].$x  -side left -fill x -padx 2p -anchor w
		}
		[set $this-port].[set $this-portsel] configure -fg #FFFFFF

		iwidgets::scrolledlistbox $childframe.listbox  -selectioncommand [format "%s ChooseMatrix" $this] -width 400p -height 300p
		set $this-matriceslistbox $childframe.listbox
		$childframe.listbox component listbox configure -listvariable $this-matrixinfotexts -selectmode browse
		pack $childframe.listbox -fill both -expand yes

		makeSciButtonPanel $w $w $this

		set matrixname [lindex [set $this-matrixname] [set $this-portsel] ]
		set selnum [lsearch [set $this-matrixnames] $matrixname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }


	}

	method SetPort {num} {
		global $this-matriceslistbox
		global $this-matrixnames
		global $this-matrixname
		global $this-port
		global $this-numport
		global $this-portsel

		
		set $this-portsel $num

		set matrixname [lindex [set $this-matrixname] [set $this-portsel] ]
		set selnum [lsearch [set $this-matrixnames] $matrixname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }
	
		for {set x 0} {$x < [set $this-numport]} {incr x} {
			[set $this-port].$x configure -fg #000000
		}
		[set $this-port].[set $this-portsel] configure -fg #FFFFFF
	
		$this ChooseMatrix
	}

	method ChooseMatrix { } {
		global $this-matriceslistbox
		global $this-matrixnames
		global $this-matrixname
		global $this-portsel
		
		set matrixnum [[set $this-matriceslistbox] curselection]
		if [expr [string equal $matrixnum ""] == 0] {
			set $this-matrixname [lreplace [set $this-matrixname] [set $this-portsel] [set $this-portsel] [lindex [set $this-matrixnames] $matrixnum] ]
		}
	}

	method ChooseFile { } {

		global env
		global $this-filename
		global $this-filename-set


		# Create a unique name for the file selection window
		set w [format "%s-filebox" .ui[modname]]

		# if the file selector is open, bring it to the front
		# in case it is iconified, deiconify
		if { [winfo exists $w] } {
	    		if { [winfo ismapped $w] == 1} {
				raise $w
	    		} else {
				wm deiconify $w
	    		}
	    		return
		}
	
		toplevel $w -class TkFDialog

		set initdir ""
	
		# place to put preferred data directory
		# it's used if $this-filename is empty
	
		# Use the standard data dirs
		# I guess there is no .mat files in there
		# at least not yet

		if {[info exists env(SCIRUN_DATA)]} {
	    		set initdir $env(SCIRUN_DATA)
		} elseif {[info exists env(SCI_DATA)]} {
	    		set initdir $env(SCI_DATA)
		} elseif {[info exists env(PSE_DATA)]} {
	    		set initdir $env(PSE_DATA)
		}
	
		iwidgets::labeledframe	$w.frame -labeltext "SELECT MATLAB FILE" 
		set childframe [$w.frame childsite]
		pack $w.frame -fill both -expand yes

		iwidgets::extfileselectionbox $childframe.fsb -mask "*.mat" -directory $initdir
		frame $childframe.bframe
		button $childframe.bframe.open -text "Open" -command "wm withdraw $w; $this OpenNewMatfile"
		button $childframe.bframe.cancel -text "Cancel" -command "wm withdraw $w"
	
		 $childframe.fsb component selection configure -textvariable $this-filename-set
			
		pack $childframe -side top -fill both -expand yes 
		pack $childframe.fsb -side top -fill both -expand yes
		pack $childframe.bframe -side top -fill x 
		pack $childframe.bframe.cancel -side left -anchor w -padx 5p -pady 5p
		pack $childframe.bframe.open -side right -anchor e -padx 5p -pady 5p
		
		wm deiconify $w	
	}
	
	method OpenNewMatfile {} {

		global $this-filename
		global $this-filename-set
		global $this-matrixnames
		global $this-matrixname
		global $this-matrixinfotexts
		global $this-portsel
		
		set $this-filename [set $this-filename-set] 
		
		# get the matrices in this file from the C++ side
		
		$this-c indexmatlabfile

		set matrixname [lindex [set $this-matrixname] [set $this-portsel] ]
		set selnum [lsearch [set $this-matrixnames] $matrixname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }
	}
	
	method OpenMatfile {} {

		global $this-filename
		global $this-matrixnames
		global $this-matrixinfotexts
		global $this-matrixname
		global $this-filename-entry
		global $this-portsel
		
		set $this-filename [[set $this-filename-entry] get] 
		
		# get the matrices in this file from the C++ side
		
		$this-c indexmatlabfile
		
		set matrixname [lindex [set $this-matrixname] [set $this-portsel] ]
		set selnum [lsearch [set $this-matrixnames] $matrixname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }
	}

}
