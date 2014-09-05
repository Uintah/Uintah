//----- PhysicalConstants.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

using namespace Uintah;

//****************************************************************************
// Default constructor 
//****************************************************************************
PhysicalConstants::PhysicalConstants():d_gravity(0,0,0), d_ref_point(0,0,0),
                                       d_viscosity(0)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
PhysicalConstants::~PhysicalConstants()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void PhysicalConstants::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PhysicalConstants");

  db->require("gravity", d_gravity); //Vector
  db->require("ref_point", d_ref_point); //IntVector
  db->require("viscosity", d_viscosity);

}
