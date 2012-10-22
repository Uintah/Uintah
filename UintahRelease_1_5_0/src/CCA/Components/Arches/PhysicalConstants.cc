/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- PhysicalConstants.cc --------------------------------------------------

#include <CCA/Components/Arches/PhysicalConstants.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/InvalidValue.h>

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

  if (params->findBlock("PhysicalConstants")) {
    ProblemSpecP db = params->findBlock("PhysicalConstants");

    db->require("gravity", d_gravity); //Vector
    db->require("reference_point", d_ref_point); //IntVector
    db->require("viscosity", d_viscosity);
  } else 
    throw InvalidValue("Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);

}
