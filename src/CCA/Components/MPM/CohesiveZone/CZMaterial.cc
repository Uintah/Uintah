/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//  CZMaterial.cc

#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <iostream>
#include <string>
#include <list>

using namespace std;
using namespace Uintah;

// Standard Constructor
CZMaterial::CZMaterial(ProblemSpecP& ps, SimulationStateP& ss,MPMFlags* flags)
  : Material(ps), d_cohesive_zone(0)
{
  d_lb = scinew MPMLabel();
  // The standard set of initializations needed
  standardInitialization(ps,flags);
  
  d_cohesive_zone = scinew CohesiveZone(this,flags,ss);
}

void
CZMaterial::standardInitialization(ProblemSpecP& ps, MPMFlags* flags)

{
  ps->require("delta_n",d_delta_n);
  ps->require("delta_t",d_delta_t);
  ps->require("sig_max",d_sig_max);
  ps->require("tau_max",d_tau_max);
  ps->require("cz_filename",d_cz_filename);
  ps->getWithDefault("do_rotation",d_do_rotation,false);
}

// Default constructor
CZMaterial::CZMaterial() : d_cohesive_zone(0)
{
  d_lb = scinew MPMLabel();
}

CZMaterial::~CZMaterial()
{
  delete d_lb;
  delete d_cohesive_zone;
}

void CZMaterial::registerParticleState(SimulationState* sharedState)
{
  sharedState->d_cohesiveZoneState.push_back(d_cohesive_zone->returnCohesiveZoneState());
  sharedState->d_cohesiveZoneState_preReloc.push_back(d_cohesive_zone->returnCohesiveZoneStatePreReloc());
}

ProblemSpecP CZMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP cz_ps = Material::outputProblemSpec(ps);
  cz_ps->appendElement("delta_n",d_delta_n);
  cz_ps->appendElement("delta_t",d_delta_t);
  cz_ps->appendElement("sig_max",d_sig_max);
  cz_ps->appendElement("tau_max",d_tau_max);
  cz_ps->appendElement("cz_filename",d_cz_filename);
  cz_ps->appendElement("do_rotation",d_do_rotation);

  return cz_ps;
}

void
CZMaterial::copyWithoutGeom(ProblemSpecP& ps,const CZMaterial* mat, 
                             MPMFlags* flags)
{
  d_delta_n = mat->d_delta_n;
  d_delta_t = mat->d_delta_t;
  d_sig_max = mat->d_sig_max;
  d_tau_max = mat->d_tau_max;
  d_cz_filename = mat->d_cz_filename;
  d_do_rotation = mat->d_do_rotation;

//  d_cohesive_zone = scinew CohesiveZone(this,flags);
}

CohesiveZone* CZMaterial::getCohesiveZone()
{
  return  d_cohesive_zone;
}

double CZMaterial::getCharLengthNormal() const
{
  return d_delta_n;
}

double CZMaterial::getCharLengthTangential() const
{
  return d_delta_t;
}

double CZMaterial::getCohesiveNormalStrength() const
{
  return d_sig_max;
}

double CZMaterial::getCohesiveTangentialStrength() const
{
  return d_tau_max;
}

string CZMaterial::getCohesiveFilename() const
{
  return d_cz_filename;
}

bool CZMaterial::getDoRotation() const
{
  return d_do_rotation;
}
