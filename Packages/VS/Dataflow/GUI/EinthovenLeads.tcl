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
#    File   : EinthovenLeads.tcl
#    Author : Martin Cole
#    Date   : Mon Mar  7 11:03:22 2005

catch {rename VS_Render_EinthovenLeads ""}

itcl_class VS_Render_EinthovenLeads {
    inherit Module
    constructor {config} {
        set name EinthovenLeads
        set_defaults
    }
    method set_defaults {} {
        global $this-lead_I
        set $this-lead_I 0

        global $this-lead_II
        set $this-lead_II 0

        global $this-lead_III
        set $this-lead_III 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	iwidgets::entryfield $w.f.options.lead_I \
	    -labeltext "Left Arm (node index):" -textvariable $this-lead_I
        pack $w.f.options.lead_I -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.lead_II \
	    -labeltext "Right Arm (node index):" -textvariable $this-lead_II
        pack $w.f.options.lead_II -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.lead_III \
	    -labeltext "Left Foot (node index):" -textvariable $this-lead_III
        pack $w.f.options.lead_III -side top -expand yes -fill x

	iwidgets::pushbutton $w.f.reset -text "Reset Data" \
	    -command "$this-c reset"
	pack $w.f.reset -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }

}
