/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Schedulers/UnstructuredDependencyException.h>
#include <sstream>

using namespace Uintah;
using namespace std;

UnstructuredDependencyException::UnstructuredDependencyException(const UnstructuredTask* task,
                                         const UnstructuredVarLabel* label,
                                               int matlIndex,
                                         const UnstructuredPatch* patch,
                                               string has,
                                               string needs,
                                         const char* file, int line)
  : task_(task), label_(label), matlIndex_(matlIndex), patch_(patch)
{
  d_msg = makeMessage( task_, label_, matlIndex_, patch_, has, needs);

#ifdef EXCEPTIONS_CRASH
  cout << "A UnstructuredDependencyException exception was thrown.\n";
  cout << file << ":" << line << "\n";
  cout << d_msg << "\n";
#endif
}

string UnstructuredDependencyException::makeMessage(const UnstructuredTask* task,
                                        const UnstructuredVarLabel* label,
                                              int matlIndex,
                                        const UnstructuredPatch* patch,
                                              string has,
                                              string needs)
{
  ostringstream str;
  str << "Task UnstructuredDependency Error: (" << has << ") has no corresponding (";
  str << needs << ") for " << label->getName();
  if (patch) {
    str << " on patch " << patch->getID();
    str << ", Level-" << patch->getLevel()->getIndex();
  }
  
  str << ", for material " << matlIndex;
  
  if (task != 0) {
    str << " in task " << task->getName();
  }
  str << ".";
  return str.str();
}

UnstructuredDependencyException::UnstructuredDependencyException(const UnstructuredDependencyException& copy)
  : task_(copy.task_),
    label_(copy.label_),
    matlIndex_(copy.matlIndex_),
    patch_(copy.patch_),
    d_msg(copy.d_msg)
{
}

const char* UnstructuredDependencyException::message() const
{
  return d_msg.c_str();
}

const char* UnstructuredDependencyException::type() const
{
  return "Uintah::Exceptions::UnstructuredDependencyException";
}

