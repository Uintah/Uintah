# 
# Makefile for sparselb testing
#

SPARSELIB_DIR = ..
include $(SPARSELIB_DIR)/makefile.def


SRCS		= tsl.cc tpre.cc tspsm.cc tsp.cc
EXECS		= tsl    tpre    tspsm    tsp
OBJS		= tsl.o  tpre.o  tspsm.o  tsp.o
all: progs

progs: $(EXECS)

clean:
	/bin/rm -f $(OBJS) out *.out

wipe: 
	/bin/rm -f $(OBJS) $(EXECS) out *.out

tsl.o : tsl.cc  
tpre.o : tpre.cc  
tspsm.o : tspsm.cc  
tsp.o : tsp.cc

tsl : tsl.o
	$(CCC) $(CCCFLAGS) -o tsl tsl.o  $(LDFLAGS)

tpre : tpre.o
	$(CCC) $(CCCFLAGS) -o tpre tpre.o  $(LDFLAGS)

tspsm : tspsm.o
	$(CCC) $(CCCFLAGS) -o tspsm tspsm.o  $(LDFLAGS)

tsp : tsp.o
	$(CCC) $(CCCFLAGS) -o tsp tsp.o  $(LDFLAGS)

