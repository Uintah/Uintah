
#ifndef Uintah_Component_Arches_CellInformation_h
#define Uintah_Component_Arches_CellInformation_h

/*
 * Placeholder - nothing here yet
 */

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Parallel/ProcessorContext.h>

namespace Uintah {
namespace Components {
  using Uintah::Interface::CFDInterface;
  using Uintah::Parallel::UintahParallelComponent;
  using Uintah::Interface::ProblemSpecP;
  using Uintah::Grid::GridP;
  using Uintah::Grid::LevelP;
  using Uintah::Grid::Patch;
  using Uintah::Interface::DataWarehouseP;
  using Uintah::Interface::SchedulerP;
  using Uintah::Parallel::ProcessorContext;


struct CellInformation {
     // p-cell geom information
     // x- direction
     Array1<double> dxep;
     Array1<double> dxpw;
     Array1<double> sew;
     // y- direction
     Array1<double> dynp;
     Array1<double> dyps;
     Array1<double> sns;
     // z-direction
     Array1<double> dztp;
     Array1<double> dzpb;
     Array1<double> stb;
     //u-cell geom info
     Array1<double> xu;
     Array1<double> dxepu;
     Array1<double> dxpwu;
     Array1<double> sewu;
     //v-cell geom info
     Array1<double> yv;
     Array1<double> dynpv;
     Array1<double> dypsv;
     Array1<double> snsv;
     //w-cell geom info
     Array1<double> zw;
     Array1<double> dztpw;
     Array1<double> dzpbw;
     Array1<double> stbw;
     //differencing factors for p-cell
     // x-direction
     Array1<double> cee;
     Array1<double> cww;
     Array1<double> cwe;
     Array1<double> ceeu;
     Array1<double> cwwu;
     Array1<double> cweu;
     Array1<double> efac;
     Array1<double> wfac;
     // y-direction
     Array1<double> cnn;
     Array1<double> css;
     Array1<double> csn;
     Array1<double> cnnv;
     Array1<double> cssv;
     Array1<double> csnv;
     Array1<double> enfac;
     Array1<double> sfac;
     // z-direction
     Array1<double> ctt;
     Array1<double> cbb;
     Array1<double> cbt;
     Array1<double> cttw;
     Array1<double> cbbw;
     Array1<double> cbtw;
     Array1<double> tfac;
     Array1<double> bfac;
     // factors for differencing u-cell
     Array1<double> fac1u;
     Array1<double> fac2u;
     Array1<double> iesdu;
     Array1<double> fac3u;
     Array1<double> fac4u;
     Array1<double> iwsdu;
     // factors for differencing v-cell
     Array1<double> fac1v;
     Array1<double> fac2v;
     Array1<double> jnsdv;
     Array1<double> fac3v;
     Array1<double> fac4v;
     Array1<double> jssdv;
     // factors for differencing w-cell
     Array1<double> fac1w;
     Array1<double> fac2w;
     Array1<double> ktsdw;
     Array1<double> fac3w;
     Array1<double> fac4w;
     Array1<double> kbsdw;
     // constructor computes the values
     CellInformation(const Patch*);
   };

} // end namespace Components
} // end namespace Uintah


#endif

