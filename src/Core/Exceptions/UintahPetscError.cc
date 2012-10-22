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


#include <Core/Exceptions/UintahPetscError.h>

#include   <sstream>

#include <iostream>

using namespace Uintah;


UintahPetscError::UintahPetscError(int petsc_code, const std::string& msg, const char* file, int line)
    : petsc_code(petsc_code)
{
  std::ostringstream out;
  out << "A UintahPetscError exception was thrown.\n"
      << file << ":" << line << "\n"
      << "PETSc error: " << petsc_code << ", " << msg;
  d_msg = out.str();
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif  
}

UintahPetscError::UintahPetscError(const UintahPetscError& copy)
    : petsc_code(copy.petsc_code), d_msg(copy.d_msg)
{
}

UintahPetscError::~UintahPetscError()
{
}

const char* UintahPetscError::message() const
{
    return d_msg.c_str();
}

const char* UintahPetscError::type() const
{
    return "Uintah::Exceptions::UintahPetscError";
}
