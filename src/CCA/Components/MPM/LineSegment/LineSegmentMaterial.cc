/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

//  LineSegmentMaterial.cc

#include <CCA/Components/MPM/LineSegment/LineSegmentMaterial.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <string>

using namespace std;
using namespace Uintah;

// Standard Constructor
LineSegmentMaterial::LineSegmentMaterial(ProblemSpecP& ps, MaterialManagerP& ss,
                                                           MPMFlags* flags)
  : Material(ps), d_linesegment(0)
{
  // The standard set of initializations needed
  standardInitialization(ps,flags);
  
  d_linesegment = scinew LineSegment(this,flags,ss);
}

void
LineSegmentMaterial::standardInitialization(ProblemSpecP& ps, MPMFlags* flags)

{
  ps->require("associated_material", d_associated_material);
  ps->require("lineseg_filename",    d_lineseg_filename);
}

// Default constructor
LineSegmentMaterial::LineSegmentMaterial() : d_linesegment(0)
{
}

LineSegmentMaterial::~LineSegmentMaterial()
{
  delete d_linesegment;
}

void LineSegmentMaterial::registerParticleState(
          std::vector<std::vector<const VarLabel*> > &LineSegmentState,
          std::vector<std::vector<const VarLabel*> > &LineSegmentState_preReloc)
{
  LineSegmentState.push_back         (d_linesegment->returnLineSegmentState());
  LineSegmentState_preReloc.push_back(d_linesegment->returnLineSegmentStatePreReloc());
}

ProblemSpecP LineSegmentMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP lineseg_ps = ps->appendChild("LineSegment");

  lineseg_ps->appendElement("associated_material", d_associated_material);
  lineseg_ps->appendElement("lineseg_filename",    d_lineseg_filename);

  return lineseg_ps;
}

void
LineSegmentMaterial::copyWithoutGeom(ProblemSpecP& ps,
                                     const LineSegmentMaterial* mat, 
                                     MPMFlags* flags)
{
  d_lineseg_filename = mat->d_lineseg_filename;
}

LineSegment* LineSegmentMaterial::getLineSegment()
{
  return  d_linesegment;
}

int LineSegmentMaterial::getAssociatedMaterial() const
{
  return d_associated_material;
}

string LineSegmentMaterial::getLineSegmentFilename() const
{
  return d_lineseg_filename;
}
