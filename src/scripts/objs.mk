ifndef _included_objs_mk
_included_objs_mk = yes

include $(OBJTOP)/scripts/targets.mk

# We don't want a parallel make if we're making in the objs directory -- that
# means we're interactive (actually programming) and want non-overlapping
# debugging output more than we want speed

ifneq (.,$(RECURSIVE_PATH))
MAKEFLAGS += -j$(MAKE_PARALLELISM)
endif

OBJS  = $(addsuffix .o,$(basename $(SRCS)))
DEPS  = $(addsuffix .d,$(basename $(SRCS)))
VPATH = $(SRCDIR)

all:: $(OBJS)

reportObjs::
	@echo $(OBJS)

_cleanHere::
	rm -f $(OBJS) $(DEPS)
	rm -fr ii_files ti_files

include $(OBJTOP)/scripts/compile.mk

-include $(DEPS)

endif
