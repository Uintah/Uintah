#
# Epilogue fragment for subdirectories.  This is only necessary if
# we are building small sos
#
# $Id$
#

ifneq ($(LARGESOS),yes)
include scripts/so_epilogue.mk
endif

#
# $Log$
# Revision 1.1  2000/03/17 09:30:57  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
