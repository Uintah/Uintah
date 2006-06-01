include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := pysci

SRCS     += 	$(SRCDIR)/api.cc \
		$(SRCDIR)/pysci_wrap.cc	

INCLUDES += -I/usr/include/python2.4

PSELIBS := 	Core/Util		\
		Core/Init		\
		Core/Basis		\
		Core/Events		\
		Core/Datatypes		\
		Core/Geom		\
		Core/Geometry		\
		Core/Exceptions		\
		Dataflow/Modules/Fields	\

LIBS := -lpython2.4

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

