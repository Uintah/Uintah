/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/OdtData.h>
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
