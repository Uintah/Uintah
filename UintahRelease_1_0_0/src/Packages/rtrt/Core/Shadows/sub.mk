
SRCDIR   := Packages/rtrt/Core/Shadows

SRCS += $(SRCDIR)/ShadowBase.cc \
	$(SRCDIR)/NoShadows.cc \
	$(SRCDIR)/HardShadows.cc \
	$(SRCDIR)/SingleSampleSoftShadows.cc \
	$(SRCDIR)/MultiSampleSoftShadows.cc \
	$(SRCDIR)/ScrewyShadows.cc \
	$(SRCDIR)/UncachedHardShadows.cc
