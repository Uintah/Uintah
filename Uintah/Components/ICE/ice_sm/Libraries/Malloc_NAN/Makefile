BIT  = 64          # 64 or n32
OBJS = AllocOS.o Allocator.o malloc.o new.o lock_mips.o

#CCFLAGS = -n32 -mips4 -Ofast=ip27 -OPT:IEEE_arithmetic=3
CCFLAGS = -I. -$(BIT) -mips4 -g -OPT:IEEE_arithmetic=3 -fullwarn
#CCFLAGS = -n32 -mips4 -g -OPT:IEEE_arithmetic=3

CCC = CC -LANG:ansi-for-init-scope=ON -MDupdate Makedepend
ASFLAGS = -$(BIT) -mips4
#ASFLAGS = -n32 -mips4
#CCFLAGS = -O -g -m486 -DNODPY
#CCC = g++

.SUFFIXES: .cc
.cc.o:
	$(CCC) -c $(CCFLAGS) $<

libMalloc_NAN.a: $(OBJS)
	rm -f $@
	ar qcv $@ $(OBJS)

include Makedepend
