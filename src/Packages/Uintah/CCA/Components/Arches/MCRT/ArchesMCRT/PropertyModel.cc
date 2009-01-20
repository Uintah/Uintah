
#include <Packages/Uintah/CCA/Components/Arches/MCRT/PropertyModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/MCRT/RadCoeff.h>
#include <Packages/Uintah/CCA/Components/Arches/MCRT/RadWsgg.h>

using namespace Uintah;
using std::cout;
using std::cerr;
using std::cin;

PropertyModel::PropertyModel(){
}


PropertyModel::~PropertyModel(){
}


void PropertyModel:: computeRadiationProps(const ProcessorGroup* pc,
					   const Patch* patch,
					   CellInformation* cellinfo,
					   ArchesVariables* vars,
					   ArchesConstVariables* constvars){
  

}
