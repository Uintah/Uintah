
# This file contains the loadToolTipText function which just fills the
# ToolTipText array with tool tips for each of the main GUI components.

proc loadToolTipText {} {

    global ToolTipText

    # Module Help texts
    set ToolTipText(Module) "L - Select\nR - Menu"
    set ToolTipText(ModuleUI) "L - Open (or raise) GUI"
    set ToolTipText(ModuleMessageBtn) "L - Open (or raise) Module Log.\nColor code: Gray -> No Message. Blue -> Informational Message.  Yellow -> Warning Message.  Red -> Error Message."
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
}

  



