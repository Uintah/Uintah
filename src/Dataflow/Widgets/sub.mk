# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Widgets

SRCS     += $(SRCDIR)/ArrowWidget.cc $(SRCDIR)/BaseWidget.cc \
	$(SRCDIR)/BoxWidget.cc $(SRCDIR)/CriticalPointWidget.cc \
	$(SRCDIR)/CrosshairWidget.cc $(SRCDIR)/FrameWidget.cc \
	$(SRCDIR)/GaugeWidget.cc $(SRCDIR)/LightWidget.cc \
	$(SRCDIR)/PathWidget.cc $(SRCDIR)/PointWidget.cc \
	$(SRCDIR)/RingWidget.cc $(SRCDIR)/ScaledBoxWidget.cc \
	$(SRCDIR)/ScaledFrameWidget.cc $(SRCDIR)/ViewWidget.cc

PSELIBS := Core/Datatypes Dataflow/Constraints Core/Exceptions \
	Core/Geom Core/Thread Core/TclInterface \
	Core/Containers Core/Geometry
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

