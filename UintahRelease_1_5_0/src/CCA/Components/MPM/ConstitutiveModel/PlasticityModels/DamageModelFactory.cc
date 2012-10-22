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


#include "DamageModelFactory.h"
#include "NullDamage.h"
#include "JohnsonCookDamage.h"
#include "HancockMacKenzieDamage.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
#include <Core/Parallel/Parallel.h>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

DamageModel* DamageModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("damage_model");
   if(!child) {
      proc0cout << "**WARNING** Creating default null damage model" << endl;
      return(scinew NullDamage());
      //throw ProblemSetupException("Cannot find damage_model tag", __FILE__, __LINE__);
   }
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for damage_model", __FILE__, __LINE__);
   
   if (mat_type == "johnson_cook")
      return(scinew JohnsonCookDamage(child));
   else if (mat_type == "hancock_mackenzie")
      return(scinew HancockMacKenzieDamage(child));
   else {
      proc0cout << "**WARNING** Creating default null damage model" << endl;
      return(scinew NullDamage(child));
      //throw ProblemSetupException("Unknown Damage Model ("+mat_type+")", __FILE__, __LINE__);
   }

   //return 0;
}

DamageModel* DamageModelFactory::createCopy(const DamageModel* dm)
{
   if (dynamic_cast<const JohnsonCookDamage*>(dm))
      return(scinew JohnsonCookDamage(dynamic_cast<const JohnsonCookDamage*>(dm)));

   else if (dynamic_cast<const HancockMacKenzieDamage*>(dm))
      return(scinew HancockMacKenzieDamage(dynamic_cast<const HancockMacKenzieDamage*>(dm)));

   else {
      proc0cout << "**WARNING** Creating copy of default null damage model" << endl;
      return(scinew NullDamage(dynamic_cast<const NullDamage*>(dm)));
      //throw ProblemSetupException("Cannot create copy of unknown damage model", __FILE__, __LINE__);
   }

   //return 0;
}
