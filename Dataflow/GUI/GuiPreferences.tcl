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

#
#  Author:          J. Davison de St. Germain
#  Creation Date:   12/18/03
#

#  This file holds procedures that are used to effect the users
#  preferences for the way certain aspects of the GUI behaves.

#  Valid settings list:
#
#     UseGuiFetch -    If on, allows the user to 'fetch' a module's GUI from anywhere on 
#                      the screen, and then return it to the pre-fetched location
#
#     MoveGuiToMouse - If on, automatically moves (most) Gui windows to a location 
#                      near the mouse (when they are initially created.)
#    

proc initGuiPreferences { } {
    global guiPreferences
    set guiPreferences("UseGuiFetch") 0
    set guiPreferences("MoveGuiToMouse") 1    
}

# Turns 'true/false', 'on/off', 'yes/no', '1/0' into '1/0' respectively
proc boolToInt { val } {
    if { $val == "true" || $val == "on" || $val == "yes" || $val == "1" } {
	return "1"
    } else {
        return "0"
    }
}

# Called from scirun_env.cc:
proc setGuiPreference { setting value } {
    global guiPreferences

    # Can't use (eg: in SciMoveToCursor.tcl) !"true"... you can only
    # use !"1"... therefore I must convert all these values to 0/1.
    set intVal [boolToInt $value]

    puts "Setting $setting to $intVal"
    if { $setting == "UseGuiFetch" } {
	set guiPreferences("UseGuiFetch") $intVal
    } elseif { $setting == "MoveGuiToMouse" } {
	set guiPreferences("MoveGuiToMouse") $intVal
    } else {
	createSciDialog -error -title "Bad Gui Preference" -button1 "Close" \
	    -message "The Gui Preference ($setting) specified in your .scirunrc file is in valid.\nIt will be ignored."
    }
}
