# TclStream.tcl
# Created by Samsonov Alexei
#   December 2000
# 

itcl_class TclStream {
    
    constructor {} {
	set strBuff ""
	set maxBuffSize 10000
	set freeBuffSize $maxBuffSize
	set txtWidget ""
	set varName ""
	set isNewOutput 0
    }

    # output text window
    public txtWidget               

    # variable serving as interface with C-part of the stream
    public varName 
 
    # holder for buffer contents               
    public strBuff                 

    public maxBuffSize  

    # set to 1 if output widget is new one, and need the buffer to purge in           
    public isNewOutput                  

    method registerVar {vname} {
	set varName $vname
	trace variable $varName w "$this flush"
	flush "" "" ""
    }
    
    method registerOutput {tname} {
	set txtWidget $tname
	set isNewOutput 1
	flush "" "" ""
    }

    method unregisterOutput {} {
	unset txtWidget
    }
    
    method flush {v ind op} {
	if {$varName!=""} {
	    
	    # no trace during execution
	    trace vdelete $varName w "$this flush"
	    set tmpBuff [set $varName]

	    if {$isNewOutput==0} {
		append strBuff $tmpBuff
		set currLength [string length $strBuff]
		
		# handling buffers of excessive size
		if {$currLength>$maxBuffSize} {
		    set strBuff [string range $strBuff [expr $currLength-$maxBuffSize/2] end]
		    set nlIndex [string first "\n" $strBuff 0]
		    set strBuff [string range $strBuff [expr $nlIndex+1] end]
		    set redraw 1
		} else {
		    set redraw 0
		}
	    }

	    if {[winfo exists $txtWidget]} {
		if {$isNewOutput} {
		    set redraw 1
		    set isNewOutput 0
		}
		
		if {$redraw} {
		    $txtWidget delete 1.0 end
		    $txtWidget insert end $strBuff
		} else {
		    $txtWidget insert end $tmpBuff
		}
	    }
	    
	    # restoring trace
	    trace variable $varName w "$this flush"
	} 
    }

}
