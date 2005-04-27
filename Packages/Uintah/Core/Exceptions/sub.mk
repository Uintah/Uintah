# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/Exceptions

SRCS     += \
	$(SRCDIR)/ConvergenceFailure.cc \
	$(SRCDIR)/InvalidCompressionMode.cc \
	$(SRCDIR)/InvalidGrid.cc            \
	$(SRCDIR)/InvalidValue.cc           \
	$(SRCDIR)/ParameterNotFound.cc      \
	$(SRCDIR)/PetscError.cc             \
	$(SRCDIR)/ProblemSetupException.cc  \
	$(SRCDIR)/TypeMismatchException.cc  \
	$(SRCDIR)/VariableNotFoundInGrid.cc \
	$(SRCDIR)/MaxIteration.cc	    

PSELIBS := \
	Core/Exceptions 

LIBS := 


