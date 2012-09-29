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

//  Material.cc

#include <Core/Grid/Material.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace Uintah;


Material::Material()
{
  thismatl=0;
  haveName = false;
  name="";
}

Material::Material(ProblemSpecP& ps)
{
  
  thismatl=0;
 
  // Look for the name attribute
  if(ps->getAttribute("name", name))
    haveName = true;
  else
    haveName = false;

}

Material::~Material()
{
  if(thismatl && thismatl->removeReference())
    delete thismatl;
}

ProblemSpecP Material::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP mat = 0;
  if (haveName) {
    mat = ps->appendChild("material");
    mat->setAttribute("name",name);
  } else {
    mat = ps->appendChild("material");
  }

  std::stringstream strstream;
  strstream << getDWIndex();
  std::string index_val = strstream.str();
  mat->setAttribute("index",index_val);
  return mat;
}

int Material::getDWIndex() const
{
  // Return this material's index into the data warehouse
  return d_dwindex;
}

void Material::setDWIndex(int idx)
{
   d_dwindex = idx;
   ASSERT(!thismatl);
   thismatl = scinew MaterialSubset(); 
                                       
   thismatl->addReference();
   thismatl->add(idx);
}

void Material::registerParticleState(SimulationState* ss)
{
}
