# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/Contact

SRCS     += $(SRCDIR)/NullContact.cc $(SRCDIR)/SingleVelContact.cc \
            $(SRCDIR)/FrictionContact.cc $(SRCDIR)/ContactFactory.cc \
	    $(SRCDIR)/Contact.cc

