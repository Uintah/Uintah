PREFIX=cube
MESH=cube_tet
NPROCS=4
UDIR ?=$(HOME)/uintah/opt
UDAVIZ ?= $(HOME)/csm-users/leavy/Codes/uda2vtk/uda2xmf.py
MESH2UINTAH ?= $(HOME)/csm-users/leavy/Codes/mesh2uintah/mesh2uintah.py

sub: 
	./submit.bash $(PREFIX)

pts:
	python $(MESH2UINTAH) $(MESH)

pfs:
	$(UDIR)/StandAlone/tools/pfs/pfs ${PREFIX}.ups

run:
	$(UDIR)/StandAlone/sus $(PREFIX).ups > Run.log

mrun:
	mpirun -np $(NNODES) $(UDIR)/StandAlone/sus -mpi $(PREFIX).ups > Run.log

viz:
	python $(UDAVIZ) $(PREFIX).uda

analyze: 
	./analyze.bash $(PREFIX)

pvviz:
	pvbatch paraview_viz.py

build: 
	$(UDIR)/build.uintah.mac
	make -j 2

wipe:
	rm -rf [0-9]* 

all: sub mviz analyze 

