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

# Makefile fragment for this subdirectory

SRCDIR := Core/GUI

SRCS := \
	$(SRCDIR)/BaseDial.tcl $(SRCDIR)/ColorPicker.tcl \
	$(SRCDIR)/Dial.tcl \
	$(SRCDIR)/Dialbox.tcl $(SRCDIR)/Doublefile.tcl \
	$(SRCDIR)/Filebox.tcl $(SRCDIR)/HelpPage.tcl \
	$(SRCDIR)/BioPSEFilebox.tcl\
	$(SRCDIR)/Histogram.tcl $(SRCDIR)/MaterialEditor.tcl \
	$(SRCDIR)/MemStats.tcl $(SRCDIR)/PointVector.tcl \
	$(SRCDIR)/ThreadStats.tcl $(SRCDIR)/Util.tcl \
	$(SRCDIR)/VTRDial.tcl\
	$(SRCDIR)/Graph.tcl \
	$(SRCDIR)/Diagram.tcl \
	$(SRCDIR)/Hairline.tcl \
	$(SRCDIR)/MinMaxWidget.tcl \
	$(SRCDIR)/OpenGLWindow.tcl\
	$(SRCDIR)/PartManager.tcl\
	$(SRCDIR)/NullGui.tcl\
#	$(SRCDIR)/GuiFilename.tcl\
#	$(SRCDIR)/Slider.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



