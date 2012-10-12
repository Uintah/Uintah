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


#include <Core/Grid/Variables/SoleVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

SoleVariableBase::~SoleVariableBase()
{
}

SoleVariableBase::SoleVariableBase()
{
}   


RefCounted*
SoleVariableBase::getRefCounted()
{
  SCI_THROW(InternalError("getRefCounted not implemented for SoleVariable", __FILE__, __LINE__));
}

const Uintah::TypeDescription* SoleVariableBase::virtualGetTypeDescription() const
{
  SCI_THROW(InternalError("virtualGetTypeDescription not implemented for SoleVariable", __FILE__, __LINE__));
}


void SoleVariableBase::emitNormal(std::ostream&, const IntVector&,
                              const IntVector&, ProblemSpecP, bool)
{
  SCI_THROW(InternalError("emitNormal not implemented for SoleVariable", __FILE__, __LINE__));

}
void SoleVariableBase::readNormal(std::istream&, bool)
{
  SCI_THROW(InternalError("readNormal not implemented for SoleVariable", __FILE__, __LINE__));

}

void SoleVariableBase::allocate(const Patch*, const IntVector&)
{
  SCI_THROW(InternalError("Should not call allocate for SoleVariable", __FILE__, __LINE__));
}

    
