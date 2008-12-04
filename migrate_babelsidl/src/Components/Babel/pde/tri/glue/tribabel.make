triIORHDRS = pde_IOR.h pde_Tri_IOR.h pdeports_FEMmatrixPort_IOR.h             \
  pdeports_IOR.h pdeports_LinSolverPort_IOR.h pdeports_MeshPort_IOR.h         \
  pdeports_PDEdescriptionPort_IOR.h pdeports_ViewPort_IOR.h
triIORSRCS = pde_Tri_IOR.c
triSKELSRCS = pde_Tri_Skel.cxx
triSTUBHDRS = pde.hxx pde_Tri.hxx pdeports.hxx pdeports_FEMmatrixPort.hxx     \
  pdeports_LinSolverPort.hxx pdeports_MeshPort.hxx                            \
  pdeports_PDEdescriptionPort.hxx pdeports_ViewPort.hxx
triSTUBSRCS = pde_Tri.cxx pdeports_FEMmatrixPort.cxx                          \
  pdeports_LinSolverPort.cxx pdeports_MeshPort.cxx                            \
  pdeports_PDEdescriptionPort.cxx pdeports_ViewPort.cxx
