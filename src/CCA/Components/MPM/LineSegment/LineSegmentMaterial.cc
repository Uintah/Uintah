/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
