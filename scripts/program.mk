ifndef _included_program_mk
_included_program_mk = yes

include $(OBJTOP)/scripts/targets.mk
include $(SRCTOP)/scripts/objs.mk

all:: $(PROGRAM)

$(PROGRAM):: $(OBJS)
	rm -f $(PROGRAM)
	$(CXX) $(LDFLAGS) -o $(PROGRAM) $(OBJS) $(LIBRARIES)

_cleanHere::
	rm -f $(PROGRAM)

endif
