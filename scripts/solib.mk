ifndef _included_solib_mk
_included_solib_mk = yes

include $(OBJTOP)/scripts/recurse.mk
include $(OBJTOP)/scripts/compile.mk

SUBOBJS = $(shell $(MAKE) --no-print-directory reportObjs)

all:: $(SOLIB)

ifeq (,$(findstring reportObjs,$(MAKECMDGOALS)))
ifeq (,$(findstring clean,$(MAKECMDGOALS)))
$(SOLIB): $(SUBOBJS)
	$(CXX) $(SOFLAGS) -o $(SOLIB) $(SUBOBJS) $(SOLIBRARIES)
endif
endif

_cleanHere::
	rm -f $(SOLIB) so_locations

endif
