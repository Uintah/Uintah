
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


# GUI for FieldReader module
# by Samsonov Alexei
# December 2000

catch {rename SCIRun_DataIO_FieldReader ""}

itcl_class SCIRun_DataIO_FieldReader {
    inherit Module

    # Multiple File Playing Timer Event ID
    protected mf_event_id -1
    protected mf_play_mode "stop"
    # 1/2 second delay
    protected mf_delay 500
    protected mf_file_list ""
    # Index of the file (in mf_file_list) to send down
    protected mf_file_number 0

    constructor {config} {
	set name FieldReader
	set_defaults
    }

    method set_defaults {} {
	global $this-types
	global $this-filetype
    }

    # Sets the first file in "filesList" to the active file, calls
    # execute, and then delays for "delay".  Repeat until "filesList"
    # is empty.
    method handleMultipleFiles { { filesList "" } { delay -1 } } {

        puts "hmf: $filesList, $delay, $mf_file_number"
        # handleMultipleFiles can be called two ways, from
        # handleMultipleFiles itself, and from the Reader dialog.  If
        # we are in the middle of a sequence and handleMultipleFiles
        # is calling itself, but the user clicks a button, then we
        # want to cancel the current event.  
        if { $mf_event_id != -1 } {
            after cancel $mf_event_id 
            set mf_event_id -1
        }

        if { $delay > 0 } {
            if { $delay < 1 } {
                puts "WARNING: casting decimal input from seconds to milliseconds!"
                # User probably put in .X seconds... translating
                set mf_delay [expr int ($delay * 1000)]
            } else {
                # Delays can only be integers...
                set mf_delay [expr int ($delay)]
            }
            puts "delaying for $mf_delay"
        }

        if { $filesList != "" } {
            set mf_file_list $filesList
            set mf_file_number 0
            puts "setting file list to $mf_file_list"
        }

        set num_files [llength $mf_file_list]
        puts "num files: $num_files"

        if { $num_files == 0 } {
            puts "error, no files specified..."
            return
        }

        puts "mode: $mf_play_mode"

        if { $mf_play_mode == "stop" } {
            return
        }

        if { $mf_play_mode == "fforward" } {
            set mf_file_number [expr $num_files - 1]
        } elseif { $mf_play_mode == "rewind" } {
            set mf_file_number 0
        } elseif { $mf_play_mode == "stepb" } {
            if { $mf_file_number == 0 || $mf_file_number == 1 } {
                set mf_file_number 0
            } else {
                incr mf_file_number -2
            }
        } elseif { $mf_play_mode == "step" } {
            if { $mf_file_number == $num_files } {
                set mf_file_number [expr $num_files - 1]
            }
        }

        # Send the current file through...
        set currentFile [lindex $mf_file_list $mf_file_number]
        incr mf_file_number

        puts "working on ([expr $mf_file_number-1]) '$currentFile'"

        set $this-filename $currentFile
        $this-c needexecute
        set remainder [lrange $filesList 1 end]

        if { $mf_play_mode == "play" && $mf_file_number != $num_files } {
            # If in play mode, then keep the sequence going...
            set mf_event_id [after $mf_delay "$this handleMultipleFiles"]
            puts "event_id: $mf_event_id"
        }
    }

    method setMultipleFilePlayMode { mode } {
        puts "setting mode to $mode"
        set mf_play_mode $mode
    }

    method ui {} {

	set w .ui[modname]

	if {[winfo exists $w]} {
	    return
	}

	toplevel $w -class TkFDialog
	# place to put preferred data directory
	# it's used if $this-filename is empty
	set initdir [netedit getenv SCIRUN_DATA]
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".fld"
	set title "Open field file"
	
	######################################################
	
	# Unwrap $this-types into a list.
	set tmp1 [set $this-types]
	set tmp2 [eval "set tmp3 $tmp1"]

	makeOpenFilebox \
		-parent $w \
		-filevar $this-filename \
	        -setcmd "wm withdraw $w" \
		-command "$this-c needexecute; wm withdraw $w" \
		-cancel "wm withdraw $w" \
		-title $title \
	        -filetypes $tmp2 \
		-initialdir $initdir \
		-defaultextension $defext \
	        -selectedfiletype $this-filetype

        # To allow for multiple files, add the line below to the above
        # call.  Note, the way multiple files are handled is by this
        # GUI sending down each file, one at a time, after a delay.
        #
        #        -allowMultipleFiles $this \

	moveToCursor $w	
    }
}
