# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Contact

SRCS     += \
	$(SRCDIR)/RigidBodyContact.cc \
	$(SRCDIR)/SingleVelContact.cc \
	$(SRCDIR)/FrictionContact.cc  \
	$(SRCDIR)/ApproachContact.cc  \
	$(SRCDIR)/ContactFactory.cc   \
	$(SRCDIR)/NullContact.cc      \
	$(SRCDIR)/Contact.cc
