##
 #  MatSelectVec.tcl: Select a row or column from a matrix
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename MatSelectVec ""}

itcl_class PSECommon_Matrix_MatSelectVec {
    inherit Module
    constructor {config} {
        set name MatSelectVec
        set_defaults
    }
    method set_defaults {} {	
	global $this-columnTCL
	global $this-columnMaxTCL
	set $this-columnTCL 0
	set $this-columnMaxTCL 100
	set $this-tcl_sel 0
        set $this-tcl_stop 0   
 }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
	set n "$this-c needexecute "

        
     	global $this-columnTCL
	global $this-columnMaxTCL
	global $this-tcl_sel
        global $this-tcl_stop


	frame $w.ff -relief groove -borderwidth 1
	radiobutton $w.ff.rb1 -text "Entire time series" -variable $this-tcl_sel -value 1  -anchor w
        radiobutton $w.ff.rb2 -text "Pick one frame" -variable $this-tcl_sel -value 0  -anchor w
	pack $w.ff.rb1 $w.ff.rb2   -fill x



	frame $w.f
        scale $w.f.s -variable $this-columnTCL \
                -from 0 -to [set $this-columnMaxTCL] \
		-label "Frame #" \
                -showvalue true -orient horizontal
	pack $w.f.s -side left -fill x -expand yes


	button $w.go -text "Execute" -relief raised -command $n 

        button $w.stop -text "Stop" -relief raised -command "$this-c stop" 
      
	pack $w.ff $w.f $w.go $w.stop -side top -fill x -expand yes

#	trace variable $this-columnMaxTCL w "$this-update"
    }

    method update {} {
	global $this-columnMaxTCL
	set w .ui[modname]
	$w.f.s config -to [set $this-columnMaxTCL]
	puts "updating!"
    }
}
