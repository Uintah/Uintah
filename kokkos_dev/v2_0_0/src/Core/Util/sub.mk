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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Util

SRCS     += \
	$(SRCDIR)/Debug.cc \
	$(SRCDIR)/DebugStream.cc \
	$(SRCDIR)/DynamicLoader.cc \
	$(SRCDIR)/DynamicCompilation.cc \
	$(SRCDIR)/ProgressReporter.cc \
	$(SRCDIR)/Endian.cc \
	$(SRCDIR)/SizeTypeConvert.cc \
	$(SRCDIR)/MacroSubstitute.cc \
	$(SRCDIR)/RCParse.cc \
	$(SRCDIR)/RWS.cc \
	$(SRCDIR)/sci_system.cc \
	$(SRCDIR)/soloader.cc \
        $(SRCDIR)/Signals.cc \
	$(SRCDIR)/Timer.cc \
	$(SRCDIR)/TypeDescription.cc \
        $(SRCDIR)/XMLParser.cc \

PSELIBS := Core/Containers Core/Exceptions Core/Thread
LIBS := $(XML_LIBRARY) $(DL_LIBRARY) $(THREAD_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

