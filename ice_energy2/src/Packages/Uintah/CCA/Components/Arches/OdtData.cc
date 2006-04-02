#include <Packages/Uintah/CCA/Components/Arches/OdtData.h>
#include <Core/Util/Endian.h>
#include <Core/Malloc/Allocator.h>

//__________________________________
  static MPI_Datatype makeMPI_odtData()
  {
//    ASSERTEQ(sizeof(odtData), sizeof(double)*10);
//    MPI_Datatype mpitype;
//    MPI_Type_vector(1, 10, 10, MPI_DOUBLE, &mpitype);
//    MPI_Type_commit(&mpitype);
//    return mpitype;
      MPI_Datatype odt_type;
      MPI_Type_contiguous(210, MPI_DOUBLE, &odt_type);
      MPI_Type_commit(&odt_type);
      return odt_type;
  }

  const Uintah::TypeDescription* Uintah::fun_getTypeDescription(Uintah::odtData*)
  {
    static Uintah::TypeDescription* td = 0;
    if(!td){
      td = scinew Uintah::TypeDescription(Uintah::TypeDescription::Other,
					  "odtData", true, 
					  &makeMPI_odtData);
    }
    return td;
  }

namespace SCIRun {

  void swapbytes( Uintah::odtData& d) {
    double *p = d.x_x;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_y;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_z;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_u;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_v;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_w;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_rho;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_T;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.x_Phi;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_u;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_v;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_w;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_rho;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_T;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.y_Phi;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_u;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_v;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_w;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_rho;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_T;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
    p = d.z_Phi;
    for (int i = 0; i < 10; i++) {
      SWAP_8(*p); 
      p++;
    }
  }
}
