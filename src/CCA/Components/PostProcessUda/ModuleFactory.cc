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

#include <CCA/Components/PostProcessUda/ModuleFactory.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <string>
#include <iostream>

using namespace Uintah;

ModuleFactory::ModuleFactory()
{
}

ModuleFactory::~ModuleFactory()
{
}

//______________________________________________________________________
//
std::vector<Module*>
ModuleFactory::create(const ProblemSpecP& prob_spec,
                      MaterialManagerP  & materialManager,
                      Output            * dataArchiver,
                      DataArchive       * dataArchive)
{
  
  std::string module("");
  
  ProblemSpecP PP_ps = prob_spec->findBlock("PostProcess");

  if( !PP_ps ) {
    throw ProblemSetupException( "\nERROR<PostProcess>: Could not find find <PostPocess> tag. \n", __FILE__, __LINE__ );
  }

  std::vector<Module*> modules;
  
  for( ProblemSpecP module_ps = PP_ps->findBlock( "Module" ); module_ps != nullptr; module_ps = module_ps->findNextBlock( "Module" ) ) {

    if( !module_ps ) {
      throw ProblemSetupException( "\nERROR<PostProcess>: Could not find find <Module> tag. \n", __FILE__, __LINE__ );
    }

    std::map<std::string, std::string> attributes;
    module_ps->getAttributes(attributes);
    module = attributes["name"];

#if 0     // Needs to be filled in.

    if ( module == "reduceUda" ) {
      // do nothing
    } 
#endif
  } 

  return modules;
}
