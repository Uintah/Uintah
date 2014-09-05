itcl_class Morgan_Readers_DenseMatrixSQLReader {
    inherit Module
    constructor {config} {
        set name DenseMatrixSQLReader
        set_defaults
    }

    method set_defaults {} {
    }

    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        global $this-database
        global $this-hostname
        global $this-port
        global $this-username
        global $this-password
        global $this-sql

        toplevel $w

        make_entry $w.database "Database:" $this-database "$this-c needexecute"
        make_entry $w.hostname "Hostname:" $this-hostname "$this-c needexecute"
        make_entry $w.port "Port:" $this-port "$this-c needexecute"
        make_entry $w.username "Username:" $this-username "$this-c needexecute"
        make_entry $w.password "Password:" $this-password "$this-c needexecute"
        make_entry $w.sql "SQL Query:" $this-sql "$this-c needexecute"
        label $w.sql_info -text "SQL Query should return (val_i1, val_i2, ... , val_in) where i is the current row number and n is the number of columns"

        pack $w.database $w.hostname $w.port $w.username $w.username $w.password $w.sql $w.sql_info -side top -padx 10 -pady 10
    }
}


