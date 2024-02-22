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

//  TriangleMaterial.cc

#include <CCA/Components/MPM/Triangle/TriangleMaterial.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <string>

using namespace std;
using namespace Uintah;

// Standard Constructor
TriangleMaterial::TriangleMaterial(ProblemSpecP& ps, MaterialManagerP& ss,
                                                           MPMFlags* flags)
  : Material(ps), d_triangle(0)
{
  d_lb = scinew MPMLabel();
  // The standard set of initializations needed
  standardInitialization(ps,flags);
  
  d_triangle = scinew Triangle(this,flags,ss);
}

void
TriangleMaterial::standardInitialization(ProblemSpecP& ps, MPMFlags* flags)

{
  ps->require("associated_material", d_associated_material);
  ps->require("triangle_filename",   d_triangle_filename);
}

// Default constructor
TriangleMaterial::TriangleMaterial() : d_triangle(0)
{
  d_lb = scinew MPMLabel();
}

TriangleMaterial::~TriangleMaterial()
{
  delete d_lb;
  delete d_triangle;
}

void TriangleMaterial::registerParticleState(
          std::vector<std::vector<const VarLabel*> > &TriangleState,
          std::vector<std::vector<const VarLabel*> > &TriangleState_preReloc)
{
  TriangleState.push_back         (d_triangle->returnTriangleState());
  TriangleState_preReloc.push_back(d_triangle->returnTriangleStatePreReloc());
}

ProblemSpecP TriangleMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP triangle_ps = ps->appendChild("Triangle");

  triangle_ps->appendElement("associated_material", d_associated_material);
  triangle_ps->appendElement("triangle_filename",   d_triangle_filename);

  return triangle_ps;
}

void
TriangleMaterial::copyWithoutGeom(ProblemSpecP& ps,
                                  const TriangleMaterial* mat, 
                                  MPMFlags* flags)
{
  d_triangle_filename = mat->d_triangle_filename;
}

Triangle* TriangleMaterial::getTriangle()
{
  return  d_triangle;
}

int TriangleMaterial::getAssociatedMaterial() const
{
  return d_associated_material;
}

string TriangleMaterial::getTriangleFilename() const
{
  return d_triangle_filename;
}
