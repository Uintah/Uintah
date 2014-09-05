/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <map>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("VarLabel", false);

static map<string, VarLabel*> allLabels;
string VarLabel::defaultCompressionMode = "none";
static Mutex lock("VarLabel create/destroy lock");

VarLabel*
VarLabel::create(const string& name,
                 const Uintah::TypeDescription* td,
                 const IntVector& boundaryLayer /*= IntVector(0,0,0) */,
                 VarType vartype /*= Normal*/)
{
  VarLabel* label = 0;
  lock.lock();
  map<string, VarLabel*>::iterator iter = allLabels.find(name);
  if(iter != allLabels.end()){
    // two labels with the same name -- make sure they are the same type
    VarLabel* dup = iter->second;
    if(boundaryLayer != dup->d_boundaryLayer)
      SCI_THROW(InternalError(string("Multiple VarLabels defined with different # of boundary layers"), __FILE__, __LINE__));

#if !defined(_AIX) && !defined(__APPLE__) && !defined(_WIN32)
    // AIX uses lib.a's, therefore the "same" var labels are different...
    // Need to look into fixing this in a better way...
    // And I am not sure why we have to do this on the mac or windows...
    if (td != dup->d_td || vartype != dup->d_vartype)
      SCI_THROW(InternalError(string("VarLabel with same name exists, '"
                          + name + "', but with different type"), __FILE__, __LINE__));
#endif
    label = dup;
  }
  else {
    label = scinew VarLabel(name, td, boundaryLayer, vartype);
    allLabels[name]=label;
    dbg << "Created VarLabel: " << label->d_name << std::endl;
  }
  label->addReference();
  lock.unlock(); 
  
  return label;
}

bool
VarLabel::destroy(const VarLabel* label)
{
  if (label == 0) return false;
  if (label->removeReference()) {
    lock.lock();    
    map<string, VarLabel*>::iterator iter = allLabels.find(label->d_name);
    if(iter != allLabels.end() && iter->second == label)
      allLabels.erase(iter); 
      
    dbg << "Deleted VarLabel: " << label->d_name << std::endl;  
    lock.unlock();
    delete label;
    
    return true;
  }

  return false;
}

VarLabel::VarLabel(const std::string& name, const Uintah::TypeDescription* td,
                   const IntVector& boundaryLayer, VarType vartype)
  : d_name(name), d_td(td), d_boundaryLayer(boundaryLayer),
    d_vartype(vartype), d_compressionMode("default"),
    d_allowMultipleComputes(false)
{
}

VarLabel::~VarLabel()
{
}

void
VarLabel::printAll()
{
  map<string, VarLabel*>::iterator iter = allLabels.begin();
  for (; iter != allLabels.end(); iter++)
    std::cerr << (*iter).second->d_name << std::endl;
}

VarLabel*
VarLabel::find(string name)
{
   map<string, VarLabel*>::iterator found = allLabels.find(name);
   if (found == allLabels.end())
      return NULL;
   else
      return found->second;
}


string
VarLabel::getFullName(int matlIndex, const Patch* patch) const
{
   ostringstream out;
        out << d_name << "(matl=" << matlIndex;
   if(patch)
        out << ", patch=" << patch->getID();
   else
        out << ", no patch";
   out << ")";
        return out.str();
}                             

void
VarLabel::allowMultipleComputes()
{
   if (!d_td->isReductionVariable())
     SCI_THROW(InternalError(string("Only reduction variables may allow multiple computes.\n'" + d_name + "' is not a reduction variable."), __FILE__, __LINE__));
   d_allowMultipleComputes = true;
}

namespace Uintah {
  ostream & 
  operator<<( ostream & out, const Uintah::VarLabel & vl )
  {
    out << vl.getName();
    return out;
  }
}

