# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Contact

SRCS     += \
	$(SRCDIR)/SpecifiedBodyContact.cc \
	$(SRCDIR)/SingleVelContact.cc \
	$(SRCDIR)/FrictionContact.cc  \
	$(SRCDIR)/ApproachContact.cc  \
	$(SRCDIR)/ContactFactory.cc   \
	$(SRCDIR)/CompositeContact.cc \
	$(SRCDIR)/NullContact.cc      \
	$(SRCDIR)/ContactMaterialSpec.cc \
	$(SRCDIR)/Contact.cc
