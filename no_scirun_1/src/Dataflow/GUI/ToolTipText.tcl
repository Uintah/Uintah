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


# This file contains the loadToolTipText function which just fills the
# ToolTipText array with tool tips for each of the main GUI components.

proc loadToolTipText {} {

    global ToolTipText

    # Module Help texts
    set ToolTipText(Module) "L - Select\nR - Menu"
    set ToolTipText(ModuleUI) "L - Open module's GUI window."
    set ToolTipText(ModuleMessageBtn) "L - Open module Log window.\nColor code:\nGray -> No Message.\nBlue -> Informational Message.\nYellow -> Warning Message.\nRed -> Error Message."
    set ToolTipText(ModuleTime) "Displays the amount of CPU time spent on this module."
    set ToolTipText(ModuleProgress) "Displays the progress of this module towards completion."
    set ToolTipText(ModulePort) "L - Highlight pipe(s) from this port (if a pipe exists).\nM - Connect port to another module (if an open port exists.)  Also displays the port's name."
    set ToolTipText(ModulePortlight) "This 'Portlight' tells the status of the port.  Black -> Off.  Red -> On.\nL - Highlight pipe(s) from this port (if a pipe exists).\nM - Connect port to another module (if an open port exists.)  Also displays the port's name."
    set ToolTipText(ModuleSubnetBtn) "L - Bring up SubNet editor."

    set ToolTipText(Connection) "L - Highlight\nCTRL-M - Delete\nR - Menu"
    set ToolTipText(Notes) "L - Edit\nM - Hide"

    set ToolTipText(FileMenu) [subst {\
       Load...          \tLoads a new network.  The current network will be lost!\n\
       Insert...        \tInserts a new network with your current network.\n\
       Save             \tSave the current dataflow network.\n\
       Save As...       \tSave the current dataflow network under a new name.\n\
       Clear Network    \tRemoves (destroys) all modules from the network editory canvas.\n\
       Execute All      \tTells the dataflow network to execute all modules.\n\
       New              \tAllows the user to create a new module.  (Mostly for code developers!)\n\
       Add Info...      \tAllows the user to annotate the current dataflow network.\n\
       Quit             \tQuits SCIRun.} ]

    set ToolTipText(HelpMenu) [subst {\
       Show Tooltips    \tToggle on/off tool tips.\n\
       About...         \tSCIRun splash screen.\n\
       License...       \tDisplays SCIRun's User License.} ]

    set ToolTipText(PackageMenus) "This menu allows you to create modules from this Package.  The modules are grouped by category."


    set ToolTipText(VCRrewind) "Jump to first frame"
    set ToolTipText(VCRstepback) "Step back one frame"
    set ToolTipText(VCRpause) "Stop"
    set ToolTipText(VCRplay) "Play"
    set ToolTipText(VCRstepforward) "Step forward one frame"
    set ToolTipText(VCRfastforward) "Jump to last frame"
    
    
}

  



