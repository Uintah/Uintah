

proc setupHelpText {} {

    global HelpText
    set HelpText(Module) "L - Select\nR - Menu"
    set HelpText(Connection) "L - Highlight\nCTRL-M - Delete\nR - Menu"
    set HelpText(Notes) "L - Edit\nM - Hide"
    
    set HelpText(FileMenu) [subst {\
       Save:               \n\
       Save As...:         \n\
       Load...             \n\
       Clear:              Removes (destroys) all modules from the network editory canvas.\n\
       Save Postscript...: \n\
       Execute All:        \n\
       New:                \n\
       Add Info...:        \n\
       Quit:               } ]
}

  


