#
# Makefile fragment for this subdirectory
# $Id$

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Uintah/Datatypes

SUBDIRS := $(SRCDIR)/Particles

include $(SRCTOP)/scripts/recurse.mk

SRCS     += $(SRCDIR)/Archive.cc  $(SRCDIR)/ArchivePort.cc \
	$(SRCDIR)/NCScalarField.cc $(SRCDIR)/CCScalarField.cc \
	$(SRCDIR)/NCVectorField.cc $(SRCDIR)/CCVectorField.cc \
	$(SRCDIR)/NCTensorField.cc $(SRCDIR)/CCTensorField.cc \
	$(SRCDIR)/TensorField.cc $(SRCDIR)/TensorFieldPort.cc \
	$(SRCDIR)/ScalarParticles.cc $(SRCDIR)/ScalarParticlesPort.cc \
	$(SRCDIR)/VectorParticles.cc $(SRCDIR)/VectorParticlesPort.cc \
	$(SRCDIR)/TensorParticles.cc $(SRCDIR)/TensorParticlesPort.cc \
	$(SRCDIR)/PSet.cc



PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers SCICore/Thread Uintah/Grid Uintah/Interface \
	PSECore/Dataflow \
        Uintah/Exceptions PSECore/XMLUtil Uintah/Components/MPM

LIBS := $(XML_LIBRARY)

ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Component/CIA Component/PIDL
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus
endif

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
# Revision 1.6  2000/12/01 20:11:54  kuzimmer
# added PSECore/Dataflow to PSELIBS so that all Port info is defined
#
# Revision 1.5  2000/10/14 03:11:30  kuzimmer
# Added code to reduce memory usage when viewing particle data
#
# Revision 1.4  2000/07/31 17:45:42  kuzimmer
# Added files and modules for Field Extraction from uda
#
# Revision 1.3  2000/06/20 20:13:27  kuzimmer
# updated so that Archiver reader will compile in Uintah
#
# Revision 1.2  2000/03/20 19:38:29  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:49  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
#
# Makefile fragment for this subdirectory
# $Id$
#
