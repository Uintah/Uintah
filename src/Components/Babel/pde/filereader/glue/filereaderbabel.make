filereaderIORHDRS = pde_IOR.h pde_PDEreader_IOR.h                             \
  pdeports_FEMmatrixPort_IOR.h pdeports_IOR.h pdeports_LinSolverPort_IOR.h    \
  pdeports_MeshPort_IOR.h pdeports_PDEdescriptionPort_IOR.h                   \
  pdeports_ViewPort_IOR.h
filereaderIORSRCS = pde_PDEreader_IOR.c
filereaderSKELSRCS = pde_PDEreader_Skel.cxx
filereaderSTUBHDRS = pde.hxx pde_PDEreader.hxx pdeports.hxx                   \
  pdeports_FEMmatrixPort.hxx pdeports_LinSolverPort.hxx pdeports_MeshPort.hxx \
  pdeports_PDEdescriptionPort.hxx pdeports_ViewPort.hxx
filereaderSTUBSRCS = pde_PDEreader.cxx pdeports_FEMmatrixPort.cxx             \
  pdeports_LinSolverPort.cxx pdeports_MeshPort.cxx                            \
  pdeports_PDEdescriptionPort.cxx pdeports_ViewPort.cxx
