#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/Endian.h>
#include <iostream>

using namespace Uintah;

SecondOrderBase::SecondOrderBase()
{
}

SecondOrderBase::~SecondOrderBase()
{
  // Destructor
}
//______________________________________________________________________
//
