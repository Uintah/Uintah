include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/CollabVis/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \

# CollabVis code begin
ifeq ($(HAVE_COLLAB_VIS),yyyyes)
  # SRCS += $(SRCDIR)/ViewServer.cc

  SUBDIRS += $(SRCDIR)/Client

  INCLUDES += -I$(SRCTOP)/$(SRCDIR)/Client

  REMOTELIBS := -L$(SRCTOP)/$(SRCDIR)/Client/lib -L$(SRCTOP)/$(SRCDIR)/Client/Network/RMF/rmf2.0/RAMP/usr/local/lib/ -L$(SRCTOP)/$(SRCDIR)/Client/Network/RMF/rmf2.0/RMF/usr/local/lib/\
	-lXML \
	-lCompression\
	-lExceptions \
	-lLogging \
	-lMessage \
	-lNetwork \
	-lProperties \
	-lRendering \
	-lThread \
        -lRAMP \
        -lRMF

  LIBS += $(XML_LIBRARY) $(REMOTELIBS) -lm
endif
# CollabVis code end

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk


