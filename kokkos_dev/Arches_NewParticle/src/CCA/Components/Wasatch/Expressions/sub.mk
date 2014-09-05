# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Wasatch/Expressions

SRCS     +=                             \
	$(SRCDIR)/BasicExprBuilder.cc	\
	$(SRCDIR)/ConvectiveFlux.cc	\
	$(SRCDIR)/DiffusiveFlux.cc	\
	$(SRCDIR)/ScalarRHS.cc		\
	$(SRCDIR)/SetCurrentTime.cc
