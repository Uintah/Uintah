//----- PhysicalConstants.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Default constructor 
//****************************************************************************
PhysicalConstants::PhysicalConstants():d_gravity(0,0,0), d_viscosity(0),
                                       d_absPressure(0)
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
  db->require("pressure", d_absPressure);
  db->require("viscosity", d_viscosity);

}

//
// $Log$
// Revision 1.6  2000/05/31 08:12:45  bbanerje
// Added Cocoon stuff to Properties, added VarLabels, changed task, requires,
// computes, get etc.in Properties, changed fixed size Mixing Var array to
// vector.
//
// Revision 1.5  2000/05/31 06:03:34  bbanerje
// Added Cocoon stuff to PhysicalConstants.h and gravity vector initializer to
// PhysicalConstants.cc
//
//

