#
# Epilogue fragment for subdirectories.  This is only necessary if
# we are building small sos
#
# $Id$
#

ifeq ($(LARGESOS),yes)
include $(SRCTOP)/scripts/so_epilogue.mk
endif

#
# $Log$
# Revision 1.2  2000/03/20 19:39:10  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:55  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
