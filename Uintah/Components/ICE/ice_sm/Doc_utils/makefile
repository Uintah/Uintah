#______________________________
# Source paths
#______________________________
DOCSOURCE =  ../main2.c \
        ../input.c \
       ../Plot_routines/plot_vector.c \
       ../Plot_routines/plot_control.c \
       ../Plot_routines/plot_face_center.c \
       ../Plot_routines/plot_common.c \
       ../Plot_routines/plot_contour.c \
       ../Plot_routines/plot_2d_line.c \
       ../Plot_routines/plot_cursor_pos.c \
       ../explicit_delPress.c\
       ../p_face.c \
       ../equate_ptr_addrss.c \
       ../interpolate_vel_CC_to_FC.c \
       ../grid.c \
       ../flux_or_primitive.c \
       ../Equation_of_state/equation_of_state.c \
       ../Equation_of_state/speed_of_sound.c \
       ../lagrangian.c \
       ../commonFunctions.c \
       ../timeadvanced.c \
       ../Advection_2D/advect_grad_limiter.c \
       ../Advection_2D/advect_centroids.c \
       ../Advection_2D/advect_preprocess.c \
       ../Advection_2D/advect_q.c \
       ../Advection_2D/advect_q_flux.c \
       ../Advection_2D/advect_q_vertex.c \
       ../Boundary_Cond/boundary_cond_FC.c \
       ../Boundary_Cond/boundary_cond.c \
       ../Write_output/output_FC.c \
       ../Write_output/output_CC.c \
       ../Write_output/output_misc.c \
       ../Source_Sinks/energy.c \
       ../Source_Sinks/momentum.c \
       ../Source_Sinks/shear_stress.c\
       ../Header_files/inline.h\
       ../initialize_variables.c \
       ../explicit_delPress.c \
       ../PressureSolve_PCG/pressure_PCG.c \
       ../PressureSolve_PCG/computeStencilWeights.c \
       ../PressureSolve_PCG/computeSource.c \
       ../PressureSolve_PCG/pressure_residual.c 
#__________________________________
#  NOW RUN the scripts
#___________________________________ 
ICE.doc: $(DOCSOURCE)
	./makedoc $(DOCSOURCE)> ICE.doc
       
ICE.html.public: $(DOCSOURCE)
	./makebody  $(DOCSOURCE)> ICE_body.html 
	./makemenu  $(DOCSOURCE)
	./makeindex $(DOCSOURCE)> ICE.html ; ./autoftp
	
       
ICE.html: $(DOCSOURCE)
	./makebody  $(DOCSOURCE)> ICE_body.html
	./makemenu  $(DOCSOURCE)
	./makeindex $(DOCSOURCE)> ICE.html

ICE.tex: $(DOCSOURCE)
	maketex $(DOCSOURCE) >ICE.tex ; latex ICE.tex

