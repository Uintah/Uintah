
catch {rename GatherTimeSteps ""}

itcl_class GatherTimeSteps {
    inherit Module
    constructor {config} {
	set name GatherTimeSteps
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w;
	    return;
	}

	set n "$this-c needexecute "

	toplevel $w
	frame $w.bot

	scale $w.s -label "Time Steps" -from 1 -to 256 -length 10c \
		-orient horizontal -variable $this-timelimit -command $n

	button $w.bot.b -text "Grab" -command "$this-c update"
	label $w.bot.l -text "Time Steps Processed: "
	label $w.bot.lv -textvariable $this-tsp

	pack $w.s $w.bot -side top -ipady 2 -ipadx 2
	pack $w.bot.b $w.bot.l $w.bot.lv -side left -ipadx 2 -ipady 2

    }
}

