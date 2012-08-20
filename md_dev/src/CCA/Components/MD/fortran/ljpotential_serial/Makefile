
FC = gfortran
FFLAGS = -g -O0
LIBS = 

SOURCES = forces.f90
OBJECTS = forces.o

.f90.o:
	$(FC) $(FFLAGS) -c $*.f

all: forces

forces: $(OBJECTS)
	$(FC) $(FFLAGS) -o forces $(OBJECTS) $(LIBS)
	
clean:
	rm -rf *.o *.mod forces

forces.o: forces.f90
	$(FC) $(FFLAGS)  -c -o $@ $<
