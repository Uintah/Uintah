#ifdef __sgi
#include <vector>
#include <set>
#include <map>

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

#include <Packages/Uintah/Core/Grid/ParticleVariableBase.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

using namespace Uintah;
using std::vector;
using std::set;
using std::map;

template class vector<Task::Dependency*>;
template class vector<const Task::Dependency*>;
template class vector<Task *>;
//template class set<VarDestType>;  // Not in .h file
//template class vector<DependData>;
//template class map< MPI_Request, DependData >;
//template class map< DependData, vector<Task *>, DependData >;
//template class map< TaskData, vector<DependData>, TaskData >;
template class vector<MPI_Request>;
template class vector<const VarLabel*>;
template class vector<vector<const VarLabel*> >;
template class vector<ParticleVariableBase*>;


template class pair<const Patch*, const Patch*>;
template class vector<ParticleSubset*>;
template class vector<const Patch*>;
template class ParticleVariable<Point>;
template class vector<int>;
//template class map<const Patch*, PatchRecord*>;
//template class map<const VarLabel*, NameRecord*, VarLabel::Compare>;
//template class 

#endif
