ifndef _included_recurse_mk
_included_recurse_mk = yes

# recurse.mk - Makefile fragment supporting recursive descent of subdirectories
#------------------------------------------------------------------------------
#
# Requires:
#
#     Must be included before targets.mk in Makefiles that use both.  Otherwise
#     recursive distclean removes the current "Makefile" before recursing.
#
#     SUBDIRS variable be set to the list of subdirectories (paths relative
#     to the directory containing the Makefile) to recursively issue make
#     commands in
#
# Provides:
#
#     RECURSIVE_PATH variable, which always contains the relative path from
#     wherever the user typed "make" to the directory containing the current
#     Makefile.  This variable is useful for printing out informative
#     messages so the user can keep track of where in the recursion errors
#     occur.
#
#     Targets "all, clean, distclean, reportObjs" (this list should match the
#     targets provided by targets.mk) will be made in each of the SUBDIRS in
#     turn.

MAKEFLAGS += --no-print-directory

all::
	@$(MAKE) RECURSIVE_TARGET=all recursive

clean::
	@$(MAKE) RECURSIVE_TARGET=clean recursive

distclean::
	@$(MAKE) RECURSIVE_TARGET=distclean recursive

reportObjs::
	@for subdir in $(SUBDIRS) ; do \
	  dirObjs="`( cd $$subdir ; $(MAKE) reportObjs )`" ; \
	  for obj in $$dirObjs ; do \
            echo $$subdir/$$obj ; \
          done ; \
	done

# The RECURSIVE_PATH make variable tracks the path from the start of a nested
# recursion.  If you type "make" in PSE, you'll get src, src/SCICore, etc.
# If you type "make" in PSE/src, you'll get SCICore, etc.  It's intended to be
# used for information purposes only.

recursive:
	@for i in $(SUBDIRS) ; do ( \
	  cd $$i ; \
	  $(MAKE) RECURSIVE_PATH=$(RECURSIVE_PATH)/$$i $(RECURSIVE_TARGET) ) ; \
	  Res=$$? ; \
	  if [ "$$Res" != "0" ] ; then \
	    exit $$Res ; \
	  fi ; \
	done

# Include targets last, so that recursive "distclean" removes the current
# Makefile after everything else is done.

include $(OBJTOP)/scripts/targets.mk

endif
