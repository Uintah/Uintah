
FC = gfortran
FFLAGS = -fd-lines-as-comments -fno-automatic -std=legacy -O3  -I$(INCLUDES)
LIB = -lgfortran

TOP = $(shell cd ../; pwd)
SRC_TOP = $(TOP)/src
EQUILIBRATION_TOP = $(TOP)/To_Run/Equilibration
COMPRESSION_TOP = $(TOP)/To_Run/Compression
INCLUDES=$(TOP)/includes

SOURCES = $(SRC_TOP)/Lucretius_Serial.f
OBJECTS = Lucretius_Serial.o


.f.o:
	$(FC) $(FFLAGS) -c $*.f

all: lucretius equilibration compression

lucretius: $(OBJECTS) $(HEADERS)
	cd $(SRC_TOP)
	$(FC) $(FFLAGS) -o Lucretius_Serial $(OBJECTS) $(LIBS)
	
equilibration: $(OBJECTS) $(HEADERS)
	cd $(EQUILIBRATION_TOP)
	$(FC) $(FFLAGS) -o $(EQUILIBRATION_TOP)/Lucretius_Serial $(OBJECTS) $(LIBS)
	
compression: $(OBJECTS) $(HEADERS)
	cd $(COMPRESSION_TOP)
	$(FC) $(FFLAGS) -o $(COMPRESSION_TOP)/Lucretius_Serial $(OBJECTS) $(LIBS)		

tags: $(SOURCES) $(HEADERS)
	ctags -w $(SOURCES) $(HEADERS)

clean:
	find $(TOP) -name "$(OBJECTS)" -exec rm -rf {} \;
	find $(TOP) -name "Lucretius_Serial" -exec rm -rf {} \;
	find $(TOP) -name "core" -exec rm -rf {} \;

Lucretius_Serial.o: Lucretius_Serial.f
	$(FC) $(FFLAGS)  -c -o $@ $<
