# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Constraints

SRCS     += $(SRCDIR)/AngleConstraint.cc $(SRCDIR)/BaseConstraint.cc \
	$(SRCDIR)/BaseVariable.cc $(SRCDIR)/CenterConstraint.cc \
	$(SRCDIR)/ConstraintSolver.cc $(SRCDIR)/DistanceConstraint.cc \
	$(SRCDIR)/LineConstraint.cc $(SRCDIR)/PlaneConstraint.cc \
	$(SRCDIR)/ProjectConstraint.cc $(SRCDIR)/PythagorasConstraint.cc \
	$(SRCDIR)/RatioConstraint.cc $(SRCDIR)/SegmentConstraint.cc \
	$(SRCDIR)/VarCore.cc

PSELIBS := Core/Containers Core/Util Core/Exceptions \
	Core/Geometry
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

