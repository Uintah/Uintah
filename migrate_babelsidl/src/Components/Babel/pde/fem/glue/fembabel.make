femIORHDRS = pde_FEM_IOR.h pde_IOR.h pdeports_FEMmatrixPort_IOR.h             \
  pdeports_IOR.h pdeports_LinSolverPort_IOR.h pdeports_MeshPort_IOR.h         \
  pdeports_PDEdescriptionPort_IOR.h pdeports_ViewPort_IOR.h
femIORSRCS = pde_FEM_IOR.c
femSKELSRCS = pde_FEM_Skel.cxx
femSTUBHDRS = pde.hxx pde_FEM.hxx pdeports.hxx pdeports_FEMmatrixPort.hxx     \
  pdeports_LinSolverPort.hxx pdeports_MeshPort.hxx                            \
  pdeports_PDEdescriptionPort.hxx pdeports_ViewPort.hxx
femSTUBSRCS = pde_FEM.cxx pdeports_FEMmatrixPort.cxx                          \
  pdeports_LinSolverPort.cxx pdeports_MeshPort.cxx                            \
  pdeports_PDEdescriptionPort.cxx pdeports_ViewPort.cxx
