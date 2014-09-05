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



itcl_class MatlabInterface_DataIO_MatlabColorMapsReader {
	inherit Module

	constructor {config} {
		set name MatlabColorMapsReader
		set_defaults
	}

	method set_defaults {} {
		global $this-filename
		global $this-colormapinfotexts
		global $this-colormapnames
		global $this-colormapname
		global $this-matriceslistbox		
		global $this-filename-set
		global $this-portsel
		global $this-numport
		global $this-disable-transpose
		
		set $this-filename ""
		set $this-colormapinfotexts ""
		set $this-colormapnames ""
		set $this-colormapname ""
		set $this-matriceslistbox ""
		set $this-filename-set ""
		set $this-portsel 0
		set $this-numport 6
		set $this-disable-transpose 0
	}


	method ui {} {

		global $this-filename
		global $this-colormapinfotexts		
		global $this-colormapnames
		global $this-colormapname
		global $this-matriceslistbox		
		global $this-filename-entry
		global $this-port
		global $this-numport
		global $this-portsel
		global $this-disable-transpose

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

		iwidgets::labeledframe $w.colormapframe -labeltext "SELECT MATLAB COLORMAP"
		set childframe [$w.colormapframe childsite]
		pack $w.colormapframe -fill both -expand yes

		frame $childframe.portframe 
		pack $childframe.portframe -fill x -pady 4p
		set $this-port $childframe.portframe
		
		for {set x 0} {$x < [set $this-numport]} {incr x} {
			button [set $this-port].$x -text [format "Port %d" [expr $x + 1]] -command [format "%s SetPort %d" $this $x]
			pack [set $this-port].$x  -side left -fill x -padx 2p -anchor w
		}
		[set $this-port].[set $this-portsel] configure -fg #FFFFFF

		iwidgets::scrolledlistbox $childframe.listbox  -selectioncommand [format "%s ChooseMatrix" $this] -width 500p -height 300p
		set $this-matriceslistbox $childframe.listbox
		$childframe.listbox component listbox configure -listvariable $this-colormapinfotexts -selectmode browse
		pack $childframe.listbox -fill both -expand yes

		frame $childframe.option
		pack $childframe.option -fill x -pady 4p
		checkbutton $childframe.option.disabletranspose -variable $this-disable-transpose -text "Disable Matlab to C++ conversion (data will be transposed)"
		pack $childframe.option.disabletranspose
		makeSciButtonPanel $w $w $this

		set colormapname [lindex [set $this-colormapname] [set $this-portsel] ]
		set selnum [lsearch [set $this-colormapnames] $colormapname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }


	}

	method SetPort {num} {
		global $this-matriceslistbox
		global $this-colormapnames
		global $this-colormapname
		global $this-port
		global $this-numport
		global $this-portsel

		
		set $this-portsel $num

		set colormapname [lindex [set $this-colormapname] [set $this-portsel] ]
		set selnum [lsearch [set $this-colormapnames] $colormapname]
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
		global $this-colormapnames
		global $this-colormapname
		global $this-portsel
		
		set colormapnum [[set $this-matriceslistbox] curselection]
		if [expr [string equal $colormapnum ""] == 0] {
			set $this-colormapname [lreplace [set $this-colormapname] [set $this-portsel] [set $this-portsel] [lindex [set $this-colormapnames] $colormapnum] ]
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
	

		makeOpenFilebox \
			-parent $w \
			-filevar $this-filename-set \
			-command "wm withdraw $w;  $this OpenNewMatfile" \
			-commandname "Open" \
			-cancel "wm withdraw $w" \
			-title "SELECT MATLAB FILE" \
			-filetypes {{ "Matlab files" "*.mat" } { "All files"  "*" } }\
			-initialdir $initdir \
			-defaultextension "*.mat" \
			-selectedfiletype 0

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
		global $this-colormapnames
		global $this-colormapname
		global $this-colormapinfotexts
		global $this-portsel
		
		set $this-filename [set $this-filename-set] 
		
		# get the matrices in this file from the C++ side
		
		$this-c indexmatlabfile

		set colormapname [lindex [set $this-colormapname] [set $this-portsel] ]
		set selnum [lsearch [set $this-colormapnames] $colormapname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }
	}
	
	method OpenMatfile {} {

		global $this-filename
		global $this-colormapnames
		global $this-colormapinfotexts
		global $this-colormapname
		global $this-filename-entry
		global $this-portsel
		
		set $this-filename [[set $this-filename-entry] get] 
		
		# get the matrices in this file from the C++ side
		
		$this-c indexmatlabfile
		
		set colormapname [lindex [set $this-colormapname] [set $this-portsel] ]
		set selnum [lsearch [set $this-colormapnames] $colormapname]
		[set $this-matriceslistbox] component listbox selection clear 0 end
		if [expr $selnum > -1] { [set $this-matriceslistbox] component listbox selection set $selnum }
	}

}
