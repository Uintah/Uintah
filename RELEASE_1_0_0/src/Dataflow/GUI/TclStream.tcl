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
    
    destructor {
	unregisterOutput
	unset varName
	destroy $this
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
	if [info exists txtWidget] {
	    unset txtWidget
	}
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
