
#ifndef Uintah_Component_Arches_CellInformation_h
#define Uintah_Component_Arches_CellInformation_h

/**************************************

CLASS
   CellInformation
   
   Short description...

GENERAL INFORMATION

   Arches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 University of Utah

KEYWORDS
   grid, nonuniform grid

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Core/Containers/OffsetArray1.h>
namespace Uintah {
  using namespace SCIRun;
  class Patch;

struct CellInformation : public RefCounted {
  // for non-uniform grid these values will come from the
  // patch but for the time being we're assigning the values 
  // locally
  OffsetArray1<double> xx;
  OffsetArray1<double> yy;
  OffsetArray1<double> zz;
  // p-cell geom information
  // x- direction
  OffsetArray1<double> dxep;
  OffsetArray1<double> dxpw;
  OffsetArray1<double> sew;
  // y- direction
  OffsetArray1<double> dynp;
  OffsetArray1<double> dyps;
  OffsetArray1<double> sns;
  // z-direction
  OffsetArray1<double> dztp;
  OffsetArray1<double> dzpb;
  OffsetArray1<double> stb;
  //u-cell geom info
  OffsetArray1<double> xu;
  OffsetArray1<double> dxepu;
  OffsetArray1<double> dxpwu;
  OffsetArray1<double> sewu;
  //v-cell geom info
  OffsetArray1<double> yv;
  OffsetArray1<double> dynpv;
  OffsetArray1<double> dypsv;
  OffsetArray1<double> snsv;
  //w-cell geom info
  OffsetArray1<double> zw;
  OffsetArray1<double> dztpw;
  OffsetArray1<double> dzpbw;
  OffsetArray1<double> stbw;
  //differencing factors for p-cell
  // x-direction
  OffsetArray1<double> cee;
  OffsetArray1<double> cww;
  OffsetArray1<double> cwe;
  OffsetArray1<double> ceeu;
  OffsetArray1<double> cwwu;
  OffsetArray1<double> cweu;
  OffsetArray1<double> efac;
  OffsetArray1<double> wfac;
  // y-direction
  OffsetArray1<double> cnn;
  OffsetArray1<double> css;
  OffsetArray1<double> csn;
  OffsetArray1<double> cnnv;
  OffsetArray1<double> cssv;
  OffsetArray1<double> csnv;
  OffsetArray1<double> enfac;
  OffsetArray1<double> sfac;
  // z-direction
  OffsetArray1<double> ctt;
  OffsetArray1<double> cbb;
  OffsetArray1<double> cbt;
  OffsetArray1<double> cttw;
  OffsetArray1<double> cbbw;
  OffsetArray1<double> cbtw;
  OffsetArray1<double> tfac;
  OffsetArray1<double> bfac;
  // factors for differencing u-cell
  OffsetArray1<double> fac1u;
  OffsetArray1<double> fac2u;
  OffsetArray1<int> iesdu;
  OffsetArray1<double> fac3u;
  OffsetArray1<double> fac4u;
  OffsetArray1<int> iwsdu;
  // factors for differencing v-cell
  OffsetArray1<double> fac1v;
  OffsetArray1<double> fac2v;
  OffsetArray1<int> jnsdv;
  OffsetArray1<double> fac3v;
  OffsetArray1<double> fac4v;
  OffsetArray1<int> jssdv;
  // factors for differencing w-cell
  OffsetArray1<double> fac1w;
  OffsetArray1<double> fac2w;
  OffsetArray1<int> ktsdw;
  OffsetArray1<double> fac3w;
  OffsetArray1<double> fac4w;
  OffsetArray1<int> kbsdw;
  // constructor computes the values
  CellInformation(const Patch*);
  ~CellInformation();
};
} // End namespace Uintah



#endif

