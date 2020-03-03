/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

//  Material.cc

#include <Core/Grid/Material.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace Uintah;


Material::Material()
{
}

Material::Material(ProblemSpecP& ps)
{
  m_matlSubset = 0;
 
  // Look for the name attribute
  if(ps->getAttribute("name", m_name))
    m_haveName = true;
  else
    m_haveName = false;
}

Material::~Material()
{
  if(m_matlSubset && m_matlSubset->removeReference())
    delete m_matlSubset;
}

ProblemSpecP Material::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP mat = 0;
  if (m_haveName) {
    mat = ps->appendChild("material");
    mat->setAttribute("name", m_name);
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
  return m_dwIndex;
}

void Material::setDWIndex(int idx)
{
   m_dwIndex = idx;
   
   ASSERT(!m_matlSubset);
   m_matlSubset = scinew MaterialSubset(); 
                                       
   m_matlSubset->addReference();
   m_matlSubset->add(m_dwIndex);
}
