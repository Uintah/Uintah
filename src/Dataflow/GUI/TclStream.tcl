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
	set varNames [list]
	set varTags [list]
    }
    
    destructor {
	unregisterOutput
	unset varNames
	destroy $this
    }

    # output text window
    public txtWidget               

    # variable serving as interface with C-part of the stream
    public varNames
    public varTags

    # holder for buffer contents               
    public strBuff                

    public maxBuffSize

    method registerVar {vname tag} {
	set index [lsearch -exact $varNames $vname]
	if {$index==-1} {
	    lappend varNames $vname
	    lappend varTags $tag
	    lsearch $varNames $vname
	    trace variable $vname w "$this flush"
	    flush $vname "" ""
	}
    }
    
    method registerOutput {tname} {
	set txtWidget $tname
	flush "" "" ""
    }

    method unregisterOutput {} {
	if [info exists txtWidget] {
	    unset txtWidget
	}
    }
    
    method flush {vname ind op} {
	if {$vname==""} {
	    # flushing the whole buffer
	    if {[winfo exists $txtWidget]} {
		$txtWidget delete 1.0 end
		$txtWidget insert end $strBuff
	    }
	} else {
	    set scName ::
	    append scName $vname
	    set index [lsearch -exact $varNames $scName]
	   
	    if {$index!=-1} {
		# no trace during execution	    
		trace vdelete $scName w "$this flush"
		set vname $scName
		set tmpBuff [set $vname]
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
		
		if {[winfo exists $txtWidget]} {
		    if {$redraw} {
			$txtWidget delete 1.0 end
			$txtWidget insert end $strBuff [lindex varTags $index]
		    } else {
			$txtWidget insert end $tmpBuff [lindex varTags $index]
		    }
		    set redraw 0
		}
		# restoring trace
		trace variable $scName w "$this flush"
	    }

	}   
    }    
}
