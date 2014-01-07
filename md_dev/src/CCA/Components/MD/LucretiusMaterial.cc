/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/MD/MDMaterial.h>
#include <CCA/Components/MD/LucretiusMaterial.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;

LucretiusMaterial::LucretiusMaterial(ProblemSpecP& ps,
                                     SimulationStateP& sharedState)
{
//  ProblemSpecP cvModel_ps = ps->findBlock("SpecificHeatModel");
//
//  ps->require("force", d_force);
//  ps->require("energy", d_energy);
//  ps->require("charge", d_charge);
//  ps->require("polarizable", d_polarizable);
//  ps->getWithDefault("polarizable", d_polarizable, true);

}

LucretiusMaterial::~LucretiusMaterial()
{

}

ProblemSpecP LucretiusMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP md_ps = Material::outputProblemSpec(ps);

//  md_ps->appendElement("force", d_force);
//  md_ps->appendElement("energy", d_energy);
//  md_ps->appendElement("charge", d_charge);
//  md_ps->appendElement("polarizability", d_polarizability);

  return md_ps;
}

void LucretiusMaterial::calculateForce()
{

}

void LucretiusMaterial::calculateEnergy()
{

}

double LucretiusMaterial::getCharge() const
{
  return d_charge;
}

double LucretiusMaterial::getPolarizability() const
{
  return d_polarizability;
}

