#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/ICE
ICE_DIR  := $(SRCTOP)/Uintah/Components/ICE/ice_sm

ifeq ($(NBITS),64)
ICE_LIBS := $(ICE_DIR)/Libraries/64bit
else
ICE_LIBS := $(ICE_DIR)/Libraries/n32bit
endif

SRCS	+= $(SRCDIR)/ICE_actual.cc $(SRCDIR)/ICE_schedule.cc \
           $(SRCDIR)/array_conversion.cc

INCLUDES += -I$(ICE_DIR)/Header_files

PSELIBS := Uintah/Interface Uintah/Grid SCICore/Exceptions 
LIBS 	:= $(XML_LIBRARY) \
           $(SRCTOP)/Uintah/Components/ICE/ice_sm/libICE.a \
           -L$(ICE_DIR) -lICE -L$(ICE_LIBS) -ltecio \
           -lcpgplot -lpgplot  -lX11 -L. -lmalloc_cv \
           -lftn -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


PROGRAM	:= $(SRCDIR)/ice
SRCS	:= $(ICE_DIR)/main2.c	\
	$(ICE_DIR)/input.c					\
	$(ICE_DIR)/Plot_routines/plot_vector.c			\
        $(ICE_DIR)/Plot_routines/plot_control.c			\
        $(ICE_DIR)/Plot_routines/plot_face_center.c		\
        $(ICE_DIR)/Plot_routines/plot_common.c			\
	$(ICE_DIR)/Plot_routines/plot_contour.c			\
        $(ICE_DIR)/Plot_routines/plot_2d_line.c			\
        $(ICE_DIR)/Plot_routines/plot_cursor_pos.c		\
        $(ICE_DIR)/p_face.c					\
        $(ICE_DIR)/explicit_delPress.c				\
        $(ICE_DIR)/equate_ptr_addrss.c				\
        $(ICE_DIR)/interpolate_vel_CC_to_FC.c			\
        $(ICE_DIR)/grid.c					\
        $(ICE_DIR)/flux_or_primitive.c				\
        $(ICE_DIR)/Equation_of_state/equation_of_state.c	\
        $(ICE_DIR)/Equation_of_state/speed_of_sound.c		\
        $(ICE_DIR)/lagrangian.c					\
        $(ICE_DIR)/commonFunctions.c				\
        $(ICE_DIR)/timeadvanced.c				\
        $(ICE_DIR)/Advection_2D/advect_grad_limiter.c		\
        $(ICE_DIR)/Advection_2D/advect_centroids.c		\
        $(ICE_DIR)/Advection_2D/advect_preprocess.c		\
        $(ICE_DIR)/Advection_2D/advect_q.c			\
        $(ICE_DIR)/Advection_2D/advect_q_flux.c			\
        $(ICE_DIR)/Advection_2D/advect_q_vertex.c		\
        $(ICE_DIR)/Boundary_Cond/boundary_cond_FC.c		\
        $(ICE_DIR)/Boundary_Cond/boundary_cond.c		\
        $(ICE_DIR)/Write_output/output_FC.c			\
        $(ICE_DIR)/Write_output/output_CC.c			\
        $(ICE_DIR)/Write_output/output_misc.c			\
        $(ICE_DIR)/Source_Sinks/energy.c			\
        $(ICE_DIR)/Source_Sinks/momentum.c			\
        $(ICE_DIR)/Source_Sinks/shear_stress.c			\
        $(ICE_DIR)/initialize_variables.c			\
        $(ICE_DIR)/nrutil+.c
LIBS:= -L$(ICE_LIBS) -ltecio -lcpgplot -lpgplot  -lX11 -L. -lmalloc_cv \
	 -lftn -lm
include $(SRCTOP)/scripts/program.mk


#
# $Log$
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


