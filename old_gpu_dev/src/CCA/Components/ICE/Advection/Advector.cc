/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/ICE/Advection/Advector.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>

using namespace std;
using namespace Uintah;

Advector::Advector()
{
    //__________________________________
    //  outflux/influx slabs
    OF_slab[RIGHT] = RIGHT;         IF_slab[RIGHT]  = LEFT;
    OF_slab[LEFT]  = LEFT;          IF_slab[LEFT]   = RIGHT;
    OF_slab[TOP]   = TOP;           IF_slab[TOP]    = BOTTOM;
    OF_slab[BOTTOM]= BOTTOM;        IF_slab[BOTTOM] = TOP;  
    OF_slab[FRONT] = FRONT;         IF_slab[FRONT]  = BACK;
    OF_slab[BACK]  = BACK;          IF_slab[BACK]   = FRONT;   

    // Slab adjacent cell
    S_ac[RIGHT]  =  IntVector( 1, 0, 0);   
    S_ac[LEFT]   =  IntVector(-1, 0, 0);   
    S_ac[TOP]    =  IntVector( 0, 1, 0);   
    S_ac[BOTTOM] =  IntVector( 0,-1, 0);   
    S_ac[FRONT]  =  IntVector( 0, 0, 1);   
    S_ac[BACK]   =  IntVector( 0, 0,-1);
}

Advector::~Advector()
{
}


//______________________________________________________________________
//  
namespace Uintah {
  //__________________________________
  void  warning_restartTimestep( vector<IntVector> badCells,
                                 vector<fflux> badOutFlux,
                                 const double vol,
                                 const int indx,
                                 const Patch* patch,
                                 DataWarehouse* new_dw)
  {
    cout << Parallel::getMPIRank() << " ERROR: ICE Advection operator: "
         << " Influx_outflux error detected, "
         << " patch " << patch->getID()
         << ", Level " << patch->getLevel()->getIndex()
         << ", matl indx "<< indx << ", number of bad cells: " << badCells.size() << endl;

    for (int i = 0; i<(int) badCells.size(); i++) {
      cout << Parallel::getMPIRank() << "  cell " <<  badCells[i] << " outflux: ";
      
      fflux& outflux_faces = badOutFlux[i];
      double total_fluxout = 0.0;
      
      for(int f = TOP; f <= BACK; f++ )  {
        double flux = outflux_faces.d_fflux[f];
        total_fluxout += flux;
        cout << " \t face: " << f << " (" << flux << ") ";
      }
      cout << " total_outflux: " << total_fluxout << endl;
    }
    
    if (new_dw->timestepRestarted() == false){
      cout << "\nA timestep restart has been requested \n " << endl;
      new_dw->restartTimestep();
    }
  }
  
  //__________________________________
  static MPI_Datatype makeMPI_fflux()
  {
    ASSERTEQ(sizeof(fflux), sizeof(double)*6);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(fflux*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "fflux", true, 
                                  &makeMPI_fflux);
    }
    return td;
  }
} // namespace Uintah


//______________________________________________________________________
//  
namespace SCIRun {

  void swapbytes( Uintah::fflux& f) {
    double *p = f.d_fflux;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }
} // namespace SCIRun
