#
# Makefile fragment for this subdirectory
# $Id$
#
#   
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/ICE

SRCS	+= $(SRCDIR)/ICE.cc       \
	$(SRCDIR)/ICELabel.cc     \
       $(SRCDIR)/ICE_schedule.cc \
       $(SRCDIR)/ICEMaterial.cc  \
       $(SRCDIR)/GeometryObject2.cc

SUBDIRS := $(SRCDIR)/EOS 
 
include $(SRCTOP)/scripts/recurse.mk          

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil Uintah/Math \
	SCICore/Datatypes

LIBS	:= $(XML_LIBRARY)        

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.20  2001/01/01 23:57:37  harman
# - Moved all scheduling of tasks over to ICE_schedule.cc
# - Added instrumentation functions
# - fixed nan's in int_eng_L_source
#
# Revision 1.19  2000/11/22 01:28:05  guilkey
# Changed the way initial conditions are set.  GeometryObjects are created
# to fill the volume of the domain.  Each object has appropriate initial
# conditions associated with it.  ICEMaterial now has an initializeCells
# method, which for now just does what was previously done with the
# initial condition stuct d_ic.  This will be extended to allow regions of
# the domain to be initialized with different materials.  Sorry for the
# lame GeometryObject2, this could be changed to ICEGeometryObject or
# something.
#
# Revision 1.18  2000/10/16 17:19:44  guilkey
# Code for ICE::step1d.  Only code for one of the faces is committed
# until things become more concrete.
#
# Revision 1.17  2000/10/06 04:05:18  jas
# Move files into EOS directory.
#
# Revision 1.16  2000/10/04 23:42:50  jas
# Add IdealGas.cc
#
# Revision 1.15  2000/10/04 20:16:37  jas
# Add EOS,Label and Material to makefile.
#
# Revision 1.14  2000/10/04 19:26:46  jas
# Changes to get ICE into UCF conformance.  Only skeleton for now.
#
# Revision 1.13  2000/07/05 22:26:19  dav
# tweaked
#
# Revision 1.12  2000/07/05 21:05:17  harman
# added icelink and ability to include testcase header files
#
# Revision 1.11  2000/07/03 16:41:46  harman
# wrapped all the steps for a single mat.  need to fix step 0 and add multimaterial
#
# Revision 1.10  2000/06/28 21:50:07  harman
# sparker, mcq, harman: - No longer need to make libICE.a in ice_sm for sus
#                       - Added iceclean to sub.mk
#                       - User now has to set environmental varialbe ICE = yes
#                       to compile the real ice code otherwise it compiles
#                       ICE_doNothing.cc
#                       - Removed the hardwired PGPLOTDIR path
#
# Revision 1.9  2000/06/28 05:15:43  sparker
# Fix the build
#
# Revision 1.8  2000/06/28 00:25:28  guilkey
# MCQ fixed this.
#
# Revision 1.7  2000/06/14 21:53:19  jehall
# - Fixed typos in last commit
#
# Revision 1.6  2000/06/14 21:37:44  jehall
# - Added generated executable 'ice' to CVS ignore list
#
# Revision 1.5  2000/06/08 02:04:08  jas
# Added stuff for making ice.
#
# Revision 1.4  2000/05/30 19:36:40  dav
# added SCICore/Exceptions to PSELIBS
#
# Revision 1.3  2000/04/12 22:58:43  sparker
# Added xerces to link line
#
# Revision 1.2  2000/03/20 19:38:21  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:30  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#


