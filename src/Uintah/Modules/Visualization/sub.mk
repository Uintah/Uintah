#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Uintah/Modules/Visualization

SRCS     += $(SRCDIR)/GridVisualizer.cc \
	$(SRCDIR)/NodeHedgehog.cc \
	$(SRCDIR)/TimestepSelector.cc \
	$(SRCDIR)/ScalarFieldExtractor.cc \
	$(SRCDIR)/VectorFieldExtractor.cc \
	$(SRCDIR)/TensorFieldExtractor.cc \
	$(SRCDIR)/ParticleFieldExtractor.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/ParticleVis.cc \
	$(SRCDIR)/EigenEvaluator.cc \
	$(SRCDIR)/InPlaneEigenEvaluator.cc \
	$(SRCDIR)/TensorElementExtractor.cc


PSELIBS :=  PSECore/Dataflow PSECore/Datatypes \
        SCICore/Thread SCICore/Persistent SCICore/Exceptions \
        SCICore/TclInterface SCICore/Containers SCICore/Datatypes \
        SCICore/Geom Uintah/Grid Uintah/Interface Uintah/Exceptions \
	SCICore/Geometry PSECore/Widgets PSECore/XMLUtil \
	SCICore/Util  Uintah/Components/MPM Uintah/Datatypes

LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
# Revision 1.7  2000/09/20 23:46:08  witzel
# Added TensorElementExtractor
#
# Revision 1.6  2000/08/25 17:27:30  witzel
# Added InPlaneEigenEvaluator
#
# Revision 1.5  2000/08/22 22:20:54  witzel
# Added EigenEvaluator
#
# Revision 1.4  2000/07/31 17:45:46  kuzimmer
# Added files and modules for Field Extraction from uda
#
# Revision 1.3  2000/06/27 16:58:00  bigler
# Added NodeHedgehog.cc
#
# Revision 1.2  2000/06/21 04:15:46  kuzimmer
# removed uneccesary dependencies on the Kurt directory
#
# Revision 1.1  2000/06/20 17:57:19  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.7  2000/06/15 19:49:39  sparker
# Link against SCICore/Util
#
# Revision 1.6  2000/06/13 20:28:16  kuzimmer
# Added a colormap key sticky for particle sets
#
# Revision 1.5  2000/06/05 21:10:31  bigler
# Added new module to visualize UINTAH grid
#
# Revision 1.4  2000/05/25 18:24:24  kuzimmer
# removing old volvis directory
#
# Revision 1.3  2000/05/24 00:25:26  kuzimmer
# Jim wants these updates so he can run Kurts stuff out of Steves 64 bit PSE. WARNING, I (dav) do not know quite what I do.
#
# Revision 1.2  2000/05/21 01:19:20  kuzimmer
# Added Archive Reader
#
# Revision 1.1  2000/05/20 02:30:09  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.4  2000/05/16 20:54:03  kuzimmer
# added new directory
#
# Revision 1.3  2000/03/21 17:33:28  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:41  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:35  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
