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



itcl_class MatlabInterface_DataIO_MatlabNrrdsWriter {
	inherit Module

	constructor {config} {
		set name MatlabNrrdsWriter
		set_defaults
	}

	method set_defaults {} {

		global $this-filename
		global $this-matrixname
		global $this-dataformat
		global $this-matrixformat		
		global $this-filename-set
		global $this-overwrite
		global $this-numport
		global $this-filename-entry
		
		set $this-filename ""
		set $this-matrixname ""
		set $this-dataformat "" 
		set $this-matrixformat ""		
		set $this-filename-set ""
		set $this-overwrite 1
		set $this-numport 6
		set $this-filename-entry ""
		
		for {set x 0} {$x < [set $this-numport]} {incr x} {
			lappend $this-matrixname [format "Nrrd%d" [expr $x + 1]]
			lappend $this-dataformat {same as data}
			lappend $this-matrixformat {numeric array}
		}
	}


	method ui {} {

		global $this-filename
		global $this-matrixname
		global $this-dataformat
		global $this-matrixformat		
		global $this-overwrite
		global $this-numport
		global $this-filename-entry
		global $this-matrixsetup
		
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
		
		label $childfile.f1.label -text ".MAT FILE "
		entry $childfile.f1.file -textvariable $this-filename -width 45
		set $this-filename-entry $childfile.f1.file  
		button $childfile.f1.browse -text "Browse" -command "$this ChooseFile"
		checkbutton $childfile.f2.overwrite -text "Confirm before overwriting an existing file" -variable $this-overwrite

		pack $childfile.f1 $childfile.f2 -fill x -expand yes
		pack $childfile.f1.label -side left -padx 3p -pady 2p -padx 4p
		pack $childfile.f1.file -side left -fill x -expand yes -padx 3p -pady 2p
		pack $childfile.f1.browse -side right -padx 3p -pady 2p -anchor e
		pack $childfile.f2.overwrite -side bottom -anchor e -padx 3p -pady 2p 

		iwidgets::labeledframe $w.matrixframe -labeltext "CREATE MATLAB MATRICES" 
		set childframe [$w.matrixframe childsite]
		set $this-matrixsetup $childframe
		
		pack $w.matrixframe -fill x -expand yes
	
		for {set x 0} {$x < [set $this-numport]} {incr x} {
			frame $childframe.port-$x -bd 2 -relief groove
			frame $childframe.port-$x.f1
			frame $childframe.port-$x.f2
			
			set matrixname [lindex [set $this-matrixname] $x]
			set dataformat [lindex [set $this-dataformat] $x]
			set matrixformat [lindex [set $this-matrixformat] $x]
			
			
			label $childframe.port-$x.f1.label -text [format "Port %d :" [expr $x+1]]
			entry $childframe.port-$x.f1.matrixname 
			$childframe.port-$x.f1.matrixname insert 0 $matrixname
			
			iwidgets::optionmenu $childframe.port-$x.f2.dataformat
			foreach dformat {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} {
				$childframe.port-$x.f2.dataformat insert end $dformat
			}
			
			set dataformatindex [lsearch {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} $dataformat]
			if [expr $dataformatindex > 0] { $childframe.port-$x.f2.dataformat select $dataformatindex }
			
			iwidgets::optionmenu $childframe.port-$x.f2.matrixformat
			foreach dformat {{numeric array} {struct array}} {
				$childframe.port-$x.f2.matrixformat insert end $dformat
			}

			set matrixformatindex [lsearch {{numeric array} {struct array}} $matrixformat]
			if [expr $matrixformatindex > 0] { $childframe.port-$x.f2.matrixformat select $matrixformatindex }
			
			pack $childframe.port-$x -fill x -expand yes -pady 5 
			pack $childframe.port-$x.f1 $childframe.port-$x.f2 -fill x -expand yes -pady 1p
			pack $childframe.port-$x.f1.label -side left -padx 2p -anchor w
			pack $childframe.port-$x.f1.matrixname -side right -fill x -expand yes -anchor n
			pack $childframe.port-$x.f2.dataformat $childframe.port-$x.f2.matrixformat -side right -padx 2p -anchor s
		}
		
		makeSciButtonPanel $w $w $this
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
	
		set $this-formatvar ""
		
		makeSaveFilebox \
			-parent $w \
			-filevar $this-filename-set \
			-command "wm withdraw $w;  $this OpenNewMatfile" \
			-commandname "Save" \
			-cancel "wm withdraw $w" \
			-title "SELECT MATLAB FILE" \
			-filetypes {{ "Matlab files" "*.mat" } }\
			-initialdir $initdir \
			-defaultextension "*.mat" \
			-selectedfiletype 0 \
			-formatvar $this-formatvar \
	        -formats {None} \
			


# CODE WAITING FOR BETTER DAYS WHEN THIS WIDGET WORKS FINE
#		iwidgets::labeledframe	$w.frame -labeltext "SELECT MATLAB FILE" 
#		set childframe [$w.frame childsite]
#		pack $w.frame -fill both -expand yes

#		iwidgets::extfileselectionbox $childframe.fsb -mask "*.mat" -directory $initdir
#		frame $childframe.bframe
#		button $childframe.bframe.open -text "Open" -command "wm withdraw $w; $this OpenNewMatfile"
#		button $childframe.bframe.cancel -text "Cancel" -command "wm withdraw $w"
	
#		 $childframe.fsb component selection configure -textvariable $this-filename-set
			
#		pack $childframe -side top -fill both -expand yes 
#		pack $childframe.fsb -side top -fill both -expand yes
#		pack $childframe.bframe -side top -fill x 
#		pack $childframe.bframe.cancel -side left -anchor w -padx 5p -pady 5p
#		pack $childframe.bframe.open -side right -anchor e -padx 5p -pady 5p

		wm deiconify $w	
	}
	
	method OpenNewMatfile {} {

		global $this-filename
		global $this-filename-set		
		set $this-filename [set $this-filename-set] 
		
	}
	
	method overwrite {} {
		global $this-overwrite
		global $this-filename
		
		if {[info exists $this-overwrite] && [info exists $this-filename] && \
			[set $this-overwrite] && [file exists [set $this-filename]] } {
				set value [tk_messageBox -type yesno -parent . \
				-icon warning -message \
				"File [set $this-filename] already exists.\n Would you like to overwrite it?"]
				if [string equal "no" $value] { return 0 }
		}
		return 1
    }
	
	method Synchronise {} {
	
		global $this-matrixname
		global $this-dataformat
		global $this-matrixformat		
		global $this-overwrite
		global $this-numport
		global $this-filename-entry
		global $this-matrixsetup
		
		set w .ui[modname]

		if {[winfo exists $w]} {

		
			set $this-filename [[set $this-filename-entry] get]
			set childframe [set $this-matrixsetup]
			set $this-matrixname ""
			set $this-dataformat ""
			set $this-matrixformat ""
		
			for {set x 0} {$x < [set $this-numport]} {incr x} {
				lappend $this-matrixname [$childframe.port-$x.f1.matrixname get] 
				lappend $this-dataformat [$childframe.port-$x.f2.dataformat get]
				lappend $this-matrixformat [$childframe.port-$x.f2.matrixformat get]
			}
		}
	}
}
