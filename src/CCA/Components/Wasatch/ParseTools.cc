/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

//-- Wasatch includes --//
#include "ParseTools.h"

//-- ExprLib includes --//
#include <expression/Tag.h>
#include <expression/ExpressionFactory.h>

#include <string>
#include <sstream>

namespace Wasatch{

  Expr::Tag
  parse_nametag( Uintah::ProblemSpecP param )
  {
    if( !param ) throw Uintah::ProblemSetupException( "NameTag not found", __FILE__, __LINE__ );
    std::string exprName;
    std::string state;
    param->getAttribute( "name", exprName );
    param->getAttribute( "state", state );
    return Expr::Tag( exprName, Expr::str2context(state) );
  }

  //============================================================================

  Category
  parse_tasklist( Uintah::ProblemSpecP param,
                  const bool isAttribute )
  {
    // jcs note that if we have a vector of attributes, then this will not work properly.
    std::string taskListName;
    if( isAttribute ){
      std::vector<std::string> tmp;
      param->getAttribute( "tasklist", tmp );
      if( tmp.size() > 1 ){
        std::ostringstream msg;
        msg << std::endl
            << "parse_tasklist() can only be used to parse single attributes, not vectors of attributes"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      assert( tmp.size() == 1 );
      taskListName = tmp[0];
    }
    else{
      param->require("TaskList",taskListName);
    }
    return select_tasklist( taskListName );
  }

  Category
  select_tasklist( const std::string& taskList )
  {
    Category cat = ADVANCE_SOLUTION;
    if     ( taskList == "initialization"   )   cat = INITIALIZATION;
    else if( taskList == "timestep_size"    )   cat = TIMESTEP_SELECTION;
    else if( taskList == "advance_solution" )   cat = ADVANCE_SOLUTION;
    else{
      std::ostringstream msg;
      msg << "ERROR: unsupported task list specified: '" << taskList << "'" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    return cat;
  }

  //============================================================================

  void
  parse_cleave_requests( Uintah::ProblemSpecP param,
                         GraphCategories& graphCat )
  {
    Expr::ExpressionFactory* const factory = graphCat[ADVANCE_SOLUTION]->exprFactory;

    for( Uintah::ProblemSpecP cleaveParams = param->findBlock("Cleave");
        cleaveParams != 0;
        cleaveParams = cleaveParams->findNextBlock("Cleave") ){

      const Expr::Tag tag = parse_nametag( cleaveParams->findBlock("NameTag") );
      std::string from;
      cleaveParams->getAttribute( "from", from );
      if( from == "PARENTS" ){
        factory->cleave_from_parents( factory->get_id(tag) );
      }
      else{
        factory->cleave_from_children( factory->get_id(tag) );
      }
    }
  }

  //============================================================================

  void
  parse_attach_dependencies( Uintah::ProblemSpecP param,
                             GraphCategories& graphCat )
  {
    for( Uintah::ProblemSpecP attachParams = param->findBlock("AttachDependency");
        attachParams != 0;
        attachParams = attachParams->findNextBlock("AttachDependency") )
    {
      const Expr::Tag src    = parse_nametag( attachParams->findBlock("Source")->findBlock("NameTag") );
      const Expr::Tag target = parse_nametag( attachParams->findBlock("Target")->findBlock("NameTag") );

      // currently we only support adding, not subtracting - this could be
      // easily changed by adding this to the parser.
      const Category cat = parse_tasklist( attachParams, true );
      graphCat[cat]->exprFactory->attach_dependency_to_expression( src, target );
    }
  }

  //============================================================================

} // namespace Wasatch
