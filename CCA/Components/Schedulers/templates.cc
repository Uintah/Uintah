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

#ifdef __sgi
#include <vector>
#include <set>
#include <map>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include <Core/Grid/Variables/ParticleVariableBase.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>


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
