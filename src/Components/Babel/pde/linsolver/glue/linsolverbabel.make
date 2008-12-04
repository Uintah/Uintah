linsolverIORHDRS = pde_IOR.h pde_LinSolver_IOR.h pdeports_FEMmatrixPort_IOR.h \
  pdeports_IOR.h pdeports_LinSolverPort_IOR.h pdeports_MeshPort_IOR.h         \
  pdeports_PDEdescriptionPort_IOR.h pdeports_ViewPort_IOR.h
linsolverIORSRCS = pde_LinSolver_IOR.c
linsolverSKELSRCS = pde_LinSolver_Skel.cxx
linsolverSTUBHDRS = pde.hxx pde_LinSolver.hxx pdeports.hxx                    \
  pdeports_FEMmatrixPort.hxx pdeports_LinSolverPort.hxx pdeports_MeshPort.hxx \
  pdeports_PDEdescriptionPort.hxx pdeports_ViewPort.hxx
linsolverSTUBSRCS = pde_LinSolver.cxx pdeports_FEMmatrixPort.cxx              \
  pdeports_LinSolverPort.cxx pdeports_MeshPort.cxx                            \
  pdeports_PDEdescriptionPort.cxx pdeports_ViewPort.cxx
