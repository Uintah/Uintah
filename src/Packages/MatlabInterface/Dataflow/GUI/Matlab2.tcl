##
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




itcl_class MatlabInterface_DataIO_Matlab2 {
    inherit Module

    constructor {config} {
        set name Matlab2
        set_defaults
    }

    method set_defaults {} {

		global $this-numport-matrix
		global $this-numport-field
		global $this-numport-nrrd

		# input matrix names

		global $this-input-matrix-name
		global $this-input-matrix-type
		global $this-input-matrix-array
		
		global $this-input-field-name
		global $this-input-field-type
		global $this-input-field-array

		global $this-input-nrrd-name
		global $this-input-nrrd-type
		global $this-input-nrrd-array

		# output matrix names

		global $this-output-matrix-name
		global $this-output-field-name
		global $this-output-nrrd-name

		# internet connection parameters

		global $this-inet-address
		global $this-inet-port
		global $this-inet-passwd
		global $this-inet-session

		# input output window

		global $this-matlab-var
		global $this-matlab-code
		global $this-matlab-output
		global $this-matlab-status

		# menu pointers
		
		global $this-matrix-menu
		global $this-field-menu
		global $this-nrrd-menu

		global $this-matlab-code-menu 
		global $this-matlab-output-menu 
		global $this-matlab-status-menu 
		global $this-matlab-add-output
		global $this-matlab-update-status

		# Set up the number of ports for each SCIRun type
		
		set $this-numport-matrix 5
		set $this-numport-field 3
		set $this-numport-nrrd 3

		# Setup the default translation options
		
		
		set $this-input-matrix-name ""
		set $this-input-matrix-type ""
		set $this-input-matrix-array ""
		set $this-output-matrix-name ""

		set $this-input-field-name ""
		set $this-input-field-array ""
		set $this-output-field-name ""

		set $this-input-nrrd-name ""
		set $this-input-nrrd-type ""
		set $this-input-nrrd-array ""
		set $this-output-nrrd-name ""
		
		for {set x 0} {$x < [set $this-numport-matrix]} {incr x} {
			lappend $this-input-matrix-name [format "i%d" [expr $x+1]]
			lappend $this-input-matrix-type {same as data}
			lappend $this-input-matrix-array {numeric array}
			lappend $this-output-matrix-name [format "o%d" [expr $x+1]]
		}

		for {set x 0} {$x < [set $this-numport-field]} {incr x} {
			lappend $this-input-field-name [format "field%d" [expr $x+1]]
			lappend $this-input-field-type {same as data}
			lappend $this-input-field-array {struct array}
			lappend $this-output-field-name [format "field%d" [expr $x+1]]
		}

		for {set x 0} {$x < [set $this-numport-nrrd]} {incr x} {
			lappend $this-input-nrrd-name [format "nrrd%d" [expr $x+1]]
			lappend $this-input-nrrd-type {same as data}
			lappend $this-input-nrrd-array {numeric array}
			lappend $this-output-nrrd-name [format "nrrd%d" [expr $x+1]]
		}

		# internet default settings
		
		set $this-inet-address "localhost"
		set $this-inet-port "5517"
		set $this-inet-passwd ""	
		set $this-inet-session "1"	

		# matlab input and output
		
		set $this-matlab-code ""
		set $this-matlab-output ""
		set $this-matlab-add-output "$this AddOutput "
		set $this-matlab-update-status "$this UpdateStatus"
		set $this-matlab-status "matlab engine not running"
	}


    method ui {} {
       
		set w .ui[modname]
		if {[winfo exists $w]} {
			raise $w
			return;
		}

		global $this-numport-matrix
		global $this-numport-field
		global $this-numport-nrrd

		# input matrix names

		global $this-input-matrix-name
		global $this-input-matrix-type
		global $this-input-matrix-array
		
		global $this-input-field-name
		global $this-input-field-array

		global $this-input-nrrd-name
		global $this-input-nrrd-type
		global $this-input-nrrd-array

		# output matrix names

		global $this-output-matrix-name
		global $this-output-field-name
		global $this-output-nrrd-name

		global $this-matrix-menu
		global $this-field-menu
		global $this-nrrd-menu
		
		global $this-matlab-code-menu 
		global $this-matlab-output-menu 
		global $this-matlab-status-menu 
	

		# internet connection parameters

		global $this-inet-address
		global $this-inet-port
		global $this-inet-passwd
		global $this-inet-session

		# input /output window

		global $this-matlab-code
		global $this-matlab-output
		global $this-matlab-var
		global $this-matlab-status

		# create a new gui window

		toplevel $w 

        wm minsize $w 100 150

		iwidgets::labeledframe $w.inetframe -labeltext "MATLAB ENGINE ADDRESS"
		set childframe [$w.inetframe childsite]
		pack $w.inetframe -fill x
		frame $childframe.f1
		frame $childframe.f2
		frame $childframe.f3
		frame $childframe.f4
		pack $childframe.f1 $childframe.f2 $childframe.f3 -side left -fill x -expand yes
		pack $childframe.f4 -side top -anchor e


		label $childframe.f1.addresslabel -text "Address:"
		entry $childframe.f1.address -textvariable $this-inet-address
		label $childframe.f2.portlabel -text "Port:" 
		entry $childframe.f2.port -textvariable $this-inet-port
		label $childframe.f3.passwdlabel -text "Password:" 
		entry $childframe.f3.passwd -textvariable $this-inet-passwd -show "*"
		label $childframe.f4.sessionlabel -text "Session:" 
		entry $childframe.f4.session -textvariable $this-inet-session


		pack $childframe.f1.addresslabel -side left -padx 3p -pady 2p -padx 4p
		pack $childframe.f1.address -side left -fill x -expand yes -padx 3p -pady 2p
		pack $childframe.f2.portlabel -side left -padx 3p -pady 2p -anchor e
		pack $childframe.f2.port -side left -padx 3p -pady 2p -anchor e
		pack $childframe.f3.passwdlabel -side left -padx 3p -pady 2p -anchor e
		pack $childframe.f3.passwd -side left -padx 3p -pady 2p -anchor e
		pack $childframe.f4.sessionlabel -side left -padx 3p -pady 2p -anchor e
		pack $childframe.f4.session -side left -padx 3p -pady 2p -anchor e


		iwidgets::labeledframe $w.ioframe -labeltext "INPUT/OUPUT"
		set childframe [$w.ioframe childsite]
		pack $w.ioframe -fill x

		iwidgets::tabnotebook $childframe.pw -height 200 -tabpos n
		$childframe.pw add -label "Matrices"
		$childframe.pw add -label "Fields" 
		$childframe.pw add -label "Nrrds" 
		$childframe.pw select 0

		pack $childframe.pw -fill x -expand yes

		set matrix [$childframe.pw childsite 0]
		set field [$childframe.pw childsite 1]
		set nrrd [$childframe.pw childsite 2]

		set $this-matrix-menu $matrix
		set $this-field-menu $field
		set $this-nrrd-menu $nrrd
		

		frame $matrix.in
		frame $matrix.out
		pack $matrix.in $matrix.out -side left -padx 5p -anchor n

		label $matrix.in.t -text "INPUT MATRICES"
		pack $matrix.in.t -side top -anchor n	
	
		label $matrix.out.t -text "OUTPUT MATRICES"
		pack $matrix.out.t -side top -anchor n	
	
		for {set x 0} {$x < [set $this-numport-matrix]} {incr x} {
	
			frame $matrix.in.m-$x
			pack $matrix.in.m-$x -side top -fill x -expand yes

			label $matrix.in.m-$x.label -text [format "matrix %d" [expr $x+1]]
			entry $matrix.in.m-$x.name 
			
			$matrix.in.m-$x.name insert 0 [lindex [set $this-input-matrix-name] $x] 
			
			iwidgets::optionmenu $matrix.in.m-$x.type
			foreach dformat {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} {
				$matrix.in.m-$x.type insert end $dformat
			}
			
			set dataformatindex [lsearch {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} [lindex [set $this-input-matrix-type] $x]]
			if [expr $dataformatindex > 0] { $matrix.in.m-$x.type select $dataformatindex }
			
			iwidgets::optionmenu $matrix.in.m-$x.array
			foreach dformat {{numeric array} {struct array}} {
				$matrix.in.m-$x.array insert end $dformat
			}

			set matrixformatindex [lsearch {{numeric array} {struct array}} [lindex [set $this-input-matrix-array] $x]]
			if [expr $matrixformatindex > 0] { $matrix.in.m-$x.array select $matrixformatindex }
			
			pack $matrix.in.m-$x.label $matrix.in.m-$x.name $matrix.in.m-$x.type $matrix.in.m-$x.array -side left

			frame $matrix.out.m-$x
			pack $matrix.out.m-$x -side top -fill x -expand yes -pady 4p
			
			label $matrix.out.m-$x.label -text [format "matrix %d" [expr $x+1]]
			entry $matrix.out.m-$x.name 

			$matrix.out.m-$x.name insert 0 [lindex [set $this-output-matrix-name] $x] 

			pack $matrix.out.m-$x.label $matrix.out.m-$x.name -side left
		}

		frame $field.in
		frame $field.out
		pack $field.in $field.out -side left -padx 5p -anchor n 

		label $field.in.t -text "INPUT FIELD MATRICES"
		pack $field.in.t -side top -anchor n	
	
		label $field.out.t -text "OUTPUT FIELD MATRICES"
		pack $field.out.t -side top -anchor n	
	
		for {set x 0} {$x < [set $this-numport-field]} {incr x} {
	
			frame $field.in.m-$x
			pack $field.in.m-$x -side top -fill x -expand yes

			label $field.in.m-$x.label -text [format "field %d" [expr $x+1]]
			entry $field.in.m-$x.name 
			
			$field.in.m-$x.name insert 0 [lindex [set $this-input-field-name] $x] 
						
			iwidgets::optionmenu $field.in.m-$x.array
			foreach dformat {{numeric array} {struct array}} {
				$field.in.m-$x.array insert end $dformat
			}

			set fieldformatindex [lsearch {{numeric array} {struct array}} [lindex [set $this-input-field-array] $x]]
			if [expr $fieldformatindex > 0] { $field.in.m-$x.array select $fieldformatindex }
			
			pack $field.in.m-$x.label $field.in.m-$x.name $field.in.m-$x.array -side left

			frame $field.out.m-$x
			pack $field.out.m-$x -side top -fill x -expand yes -pady 4p
			
			label $field.out.m-$x.label -text [format "field %d" [expr $x+1]]
			entry $field.out.m-$x.name 

			$field.out.m-$x.name insert 0 [lindex [set $this-output-field-name] $x] 

			pack $field.out.m-$x.label $field.out.m-$x.name -side left
		}


		frame $nrrd.in
		frame $nrrd.out
		pack $nrrd.in $nrrd.out -side left -padx 5p -anchor n 

		label $nrrd.in.t -text "INPUT NRRD MATRICES"
		pack $nrrd.in.t -side top -anchor n	
	
		label $nrrd.out.t -text "OUTPUT NRRD MATRICES"
		pack $nrrd.out.t -side top -anchor n	
	
		for {set x 0} {$x < [set $this-numport-nrrd]} {incr x} {
	
			frame $nrrd.in.m-$x
			pack $nrrd.in.m-$x -side top -fill x -expand yes

			label $nrrd.in.m-$x.label -text [format "nrrd %d" [expr $x+1]]
			entry $nrrd.in.m-$x.name 
			
			$nrrd.in.m-$x.name insert 0 [lindex [set $this-input-nrrd-name] $x] 
			
			iwidgets::optionmenu $nrrd.in.m-$x.type
			foreach dformat {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} {
				$nrrd.in.m-$x.type insert end $dformat
			}
			
			set dataformatindex [lsearch {{same as data} {double} {single} {int8} {uint8} {int16} {uint16} {int32} {uint32}} [lindex [set $this-input-nrrd-type] $x]]
			if [expr $dataformatindex > 0] { $nrrd.in.m-$x.type select $dataformatindex }
			
			iwidgets::optionmenu $nrrd.in.m-$x.array
			foreach dformat {{numeric array} {struct array}} {
				$nrrd.in.m-$x.array insert end $dformat
			}

			set nrrdformatindex [lsearch {{numeric array} {struct array}} [lindex [set $this-input-nrrd-array] $x]]
			if [expr $nrrdformatindex > 0] { $nrrd.in.m-$x.array select $nrrdformatindex }
			
			pack $nrrd.in.m-$x.label $nrrd.in.m-$x.name $nrrd.in.m-$x.type $nrrd.in.m-$x.array -side left

			frame $nrrd.out.m-$x
			pack $nrrd.out.m-$x -side top -fill x -expand yes -pady 4p
			
			label $nrrd.out.m-$x.label -text [format "nrrd %d" [expr $x+1]]
			entry $nrrd.out.m-$x.name 

			$nrrd.out.m-$x.name insert 0 [lindex [set $this-output-nrrd-name] $x] 

			pack $nrrd.out.m-$x.label $nrrd.out.m-$x.name -side left
		}

		iwidgets::labeledframe $w.matlabframe -labeltext "MATLAB"
		set childframe [$w.matlabframe childsite]
		pack $w.matlabframe -fill both -expand yes
		
		iwidgets::tabnotebook $childframe.pw -tabpos n -height 300
		$childframe.pw add -label "Matlab Code"
		$childframe.pw add -label "Matlab Engine Output" 
		$childframe.pw add -label "Matlab Engine Status" 
		$childframe.pw select 0

		pack $childframe.pw -fill both -expand yes
		
		set $this-matlab-code-menu [$childframe.pw childsite 0]
		set $this-matlab-output-menu [$childframe.pw childsite 1]
		set $this-matlab-status-menu [$childframe.pw childsite 2]
		set code [$childframe.pw childsite 0]
		set output [$childframe.pw childsite 1]
		set status [$childframe.pw childsite 2]
		
		frame $code.f1
		frame $output.f1
		frame $status.f1
		frame $code.f2
		frame $output.f2
		frame $status.f2
		pack $code.f1 $output.f1 $status.f1 -side top -fill both -expand yes
		pack $code.f2 $output.f2 $status.f2 -side top -fill x  

		option add *textBackground white	
		iwidgets::scrolledtext $code.f1.cmd -vscrollmode dynamic \
			-labeltext "Matlab Commands" -height 150 
		bind $code.f1.cmd <Leave> "$this update_text"
		$code.f1.cmd insert end [set $this-matlab-code]
		pack $code.f1.cmd -fill both -expand yes
		button $code.f2.clear -text "clear" -command "$this ClearMCode"
		pack $code.f2.clear -anchor e 


		iwidgets::scrolledtext $output.f1.display -vscrollmode dynamic \
			-labeltext "Matlab Output" -height 150 
		$output.f1.display clear	
		$output.f1.display insert end [set $this-matlab-output]
		set $this-matlab-var $this-matlab-output
		pack $output.f1.display -fill both -expand yes
		button $output.f2.clear -text "clear" -command "$this ClearOutput"
		pack $output.f2.clear -anchor e 

		iwidgets::scrolledtext $status.f1.status -vscrollmode dynamic \
		    -labeltext "Matlab Engine Information" -height 150
		$status.f1.status clear	
		$status.f1.status insert end [set $this-matlab-status]
		pack $status.f1.status -fill both -expand yes

		makeSciButtonPanel $w $w $this
    }

	method UpdateStatus {text} {
	
		global $this-matlab-status-menu
		global $this-matlab-status
		
		set $this-matlab-status $text
		
		set w .ui[modname]
		if {[winfo exists $w]} {
			set menu [set $this-matlab-status-menu]
			$menu.f1.status clear
			$menu.f1.status insert end $text
			return;
		}
		
	}

	method ClearOutput {} {
		global $this-matlab-output-menu
		global $this-matlab-output
		
		set $this-matlab-output ""
		set w .ui[modname]
		if {[winfo exists $w]} {
			set menu [set $this-matlab-output-menu]
			$menu.f1.display clear
			return;
		}
	}

	method ClearMCode {} {
		global $this-matlab-code-menu
		global $this-matlab-code
		
		set $this-matlab-code ""
		set w .ui[modname]
		if {[winfo exists $w]} {
			set menu [set $this-matlab-code-menu]
			$menu.f1.display clear
			return;
		}
	}


	method AddOutput {text} {
		global $this-matlab-output-menu
		global $this-matlab-output
		
		append $this-matlab-output $text
		set w .ui[modname]
		if {[winfo exists $w]} {
			set menu [set $this-matlab-output-menu]
			$menu.f1.display insert end $text
			return;
		}
	}

	method Synchronise {} {
	
		global $this-input-matrix-name
		global $this-input-matrix-type
		global $this-input-matrix-array
		global $this-output-matrix-name

		global $this-input-field-name
		global $this-input-field-array
		global $this-output-field-name

		global $this-input-nrrd-name
		global $this-input-nrrd-type
		global $this-input-nrrd-array
		global $this-output-nrrd-name
		
		global $this-numport-matrix
		global $this-numport-nrrd
		global $this-numport-field

		global $this-matrix-menu
		global $this-nrrd-menu
		global $this-field-menu
		
		set w .ui[modname]

		if {[winfo exists $w]} {

			set $this-input-matrix-name ""
			set $this-input-matrix-type ""
			set $this-input-matrix-array ""
			set $this-output-matrix-name ""
			
			set matrix [set $this-matrix-menu]
			set field [set $this-field-menu]
			set nrrd [set $this-nrrd-menu]
		
			for {set x 0} {$x < [set $this-numport-matrix]} {incr x} {
				lappend $this-input-matrix-name [$matrix.in.m-$x.name get] 
				lappend $this-input-matrix-type [$matrix.in.m-$x.type get] 
				lappend $this-input-matrix-array [$matrix.in.m-$x.array get] 
				lappend $this-output-matrix-name [$matrix.out.m-$x.name get] 
			}

			set $this-input-field-name ""
			set $this-input-field-array ""
			set $this-output-field-name ""
		
			for {set x 0} {$x < [set $this-numport-field]} {incr x} {
				lappend $this-input-field-name [$field.in.m-$x.name get] 
				lappend $this-input-field-array [$field.in.m-$x.array get] 
				lappend $this-output-field-name [$field.out.m-$x.name get] 
			}

			set $this-input-nrrd-name ""
			set $this-input-nrrd-type ""
			set $this-input-nrrd-array ""
			set $this-output-nrrd-name ""
		
			for {set x 0} {$x < [set $this-numport-nrrd]} {incr x} {
				lappend $this-input-nrrd-name [$nrrd.in.m-$x.name get] 
				lappend $this-input-nrrd-type [$nrrd.in.m-$x.type get] 
				lappend $this-input-nrrd-array [$nrrd.in.m-$x.array get] 
				lappend $this-output-nrrd-name [$nrrd.out.m-$x.name get] 
			}

			
		}
	}

    method update_text {} {
	set w .ui[modname]
	global $this-matlab-code-menu
	set code [set $this-matlab-code-menu]
	set $this-matlab-code [$code.f1.cmd get 1.0 end]
    }

}
