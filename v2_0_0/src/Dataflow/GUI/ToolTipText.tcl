
# This file contains the loadToolTipText function which just fills the
# ToolTipText array with tool tips for each of the main GUI components.

proc loadToolTipText {} {

    global ToolTipText

    # Module Help texts
    set ToolTipText(Module) "L - Select\nR - Menu"
    set ToolTipText(ModuleUI) "L - Open (or raise) GUI"
    set ToolTipText(ModuleMessageBtn) "L - Open (or raise) Module Log"
    set ToolTipText(ModuleTime) "Displays the amount of CPU time spent on this module."
    set ToolTipText(ModuleProgress) "Displays the progress of this module towards completion."
    set ToolTipText(ModulePort) "L - Highlight\nM - Display port name"

    set ToolTipText(Connection) "L - Highlight\nCTRL-M - Delete\nR - Menu"
    set ToolTipText(Notes) "L - Edit\nM - Hide"

    set ToolTipText(FileMenu) [subst {\
       Save                Save the current dataflow network.\n\
       Save As...          Save the current dataflow network under a new name.\n\
       Load...             Loads a new network.  The current network will be lost!\n\
       Insert...           Inserts a new network with your current network.\n\
       Clear               Removes (destroys) all modules from the network editory canvas.\n\
       Execute All         Tells the dataflow network to execute all modules.\n\
       New                 Allows the user to create a new module.  (Mostly for code developers!)\n\
       Add Info...         Allows the user to annotate the current dataflow network.\n\
       Quit                Quits SCIRun.} ]
}

  



