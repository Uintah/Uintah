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

//--- Local (Wasatch) includes ---//
#include "Properties.h"
#include "GraphHelperTools.h"
#include "ParseTools.h"
#include "FieldAdaptor.h"
#include "Expressions/TabPropsEvaluator.h"
#include "Expressions/DensityCalculator.h"
#include "Expressions/SolnVarEst.h"
#include "TagNames.h"

//--- ExprLib includes ---//
#include <expression/ExpressionFactory.h>

//--- TabProps includes ---//
#include <tabprops/StateTable.h>

//--- Uintah includes ---//
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <fstream>

using std::endl;
using std::flush;

#ifndef TabProps_BSPLINE
// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;
#endif

namespace Wasatch{

  //====================================================================

  /**
   *  \ingroup WasatchParser
   *  \brief set up TabProps for use on the given GraphHelper
   *  \param params - the parser parameters for the TabProps specification.
   *  \param gh - the GraphHelper associated with this instance of TabProps.
   *  \param cat - the Category specifying the task associated with this 
   *         instance of TabProps.
   *  \param doDenstPlus - the boolean showing wether we have a variable 
   *         density case and we want to do pressure projection or not
   */
  void parse_tabprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh,
                       Category cat,
                       const bool doDenstPlus ) 
  {
    std::string fileName;
    params->get("FileNamePrefix",fileName);

    proc0cout << "Loading TabProps file '" << fileName << "' ... " << std::flush;

    StateTable table;
    try{
      table.read_table( fileName+".tbl" );
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << e.what() << std::endl << std::endl
          << "Error reading TabProps file '" << fileName << ".tbl'" << std::endl
          << "Check to ensure that the file exists." << std::endl
          << "It is also possible that there was an error loading the file.  This could be caused" << std::endl
          << "by a file in a format incompatible with the version of TabProps linked in here." << std::endl
          << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    proc0cout << "done" << std::endl;

    //___________________________________________________________
    // get information for the independent variables in the table
    typedef std::map<std::string,Expr::Tag> VarNameMap;
    VarNameMap ivarMap;

    for( Uintah::ProblemSpecP ivarParams = params->findBlock("IndependentVariable");
         ivarParams != 0;
         ivarParams = ivarParams->findNextBlock("IndependentVariable") ){
      std::string ivarTableName;
      const Expr::Tag ivarTag = parse_nametag( ivarParams->findBlock("NameTag") );
      ivarParams->get( "NameInTable", ivarTableName );
      ivarMap[ivarTableName] = ivarTag;
    }

    //______________________________________________________________
    // NOTE: the independent variable names must be specified in the
    // exact order dictated by the table.  This order will determine
    // the ordering for the arguments to the evaluator later on.
    typedef std::vector<std::string> Names;
    std::vector<Expr::Tag> ivarNames;
    const Names& ivars = table.get_indepvar_names();
    for( Names::const_iterator inm=ivars.begin(); inm!=ivars.end(); ++inm ){
      ivarNames.push_back( ivarMap[*inm] );
    }

    //________________________________________________________________
    // create an expression for each property.  alternatively, we
    // could create an expression that evaluated all required
    // properties at once, since the expression has that capability...
    for( Uintah::ProblemSpecP dvarParams = params->findBlock("ExtractVariable");
         dvarParams != 0;
         dvarParams = dvarParams->findNextBlock("ExtractVariable") ){

      //_______________________________________
      // extract dependent variable information
      const Expr::Tag dvarTag = parse_nametag( dvarParams->findBlock("NameTag") );

      std::string dvarTableName;
      dvarParams->get( "NameInTable", dvarTableName );
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarTableName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      proc0cout << "Constructing property evaluator for '" << dvarTag
                << "' from file '" << fileName << "'." << std::endl;
      const InterpT* const interp = table.find_entry( dvarTableName );
      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SVolField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case XVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfXField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case YVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfYField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case ZVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfZField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type named '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
        throw std::runtime_error( msg.str() );
      }
    }

    //________________________________________________________________
    // create an expression specifically for density.

    for( Uintah::ProblemSpecP densityParams = params->findBlock("ExtractDensity");
         densityParams != 0;
         densityParams = densityParams->findNextBlock("ExtractDensity") ){
        
      std::vector<Expr::Tag> rhoEtaTags;
      std::vector<Expr::Tag> reiEtaTags;
      for( Uintah::ProblemSpecP rhoEtaParams = densityParams->findBlock("DensityWeightedIVar");
          rhoEtaParams != 0;
          rhoEtaParams = rhoEtaParams->findNextBlock("DensityWeightedIVar") ){
        const Expr::Tag rhoEtaTag = parse_nametag( rhoEtaParams->findBlock("NameTag") );
        rhoEtaTags.push_back( rhoEtaTag );
        Uintah::ProblemSpecP reiEtaParams = rhoEtaParams->findBlock("RelatedIVar");
        const Expr::Tag reiEtaTag = parse_nametag( reiEtaParams->findBlock("NameTag") );
        reiEtaTags.push_back( reiEtaTag );
      }


      //_______________________________________
      // extract density variable information
      std::string dvarTableName;
      densityParams->get( "NameInTable", dvarTableName );
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarTableName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      const InterpT* const interp = table.find_entry( dvarTableName );

      //_____________________________________
      // register the expression for density
      Expr::Tag densityTag = parse_nametag( densityParams->findBlock("NameTag") );
      typedef DensityCalculator<SpatialOps::structured::SVolField>::Builder DensCalc;
      gh.exprFactory->register_expression( scinew DensCalc( densityTag, interp->clone(), rhoEtaTags,
                                                            reiEtaTags, ivarNames ) );

      //_________________________________________________________________________________________
      // preparing the input arguments of the expression for density at the next RK time stages if they are needed to be estimated
      Expr::Tag densityStarTag = Expr::Tag();
      Expr::TagList rhoEtaStarTags, ivarStarTags, reiEtaStarTags;

      Expr::Tag density2StarTag = Expr::Tag();
      Expr::TagList rhoEta2StarTags, ivar2StarTags, reiEta2StarTags;

      if (doDenstPlus && cat==INITIALIZATION) {    
        const TagNames& tagNames = TagNames::self();
        densityStarTag = Expr::Tag(densityTag.name() + tagNames.star, Expr::STATE_NONE);
        density2StarTag = Expr::Tag(densityTag.name() + tagNames.doubleStar, Expr::STATE_NONE);
        typedef DensityCalculator<SpatialOps::structured::SVolField>::Builder DensCalc;
        const Expr::ExpressionID densStarID = gh.exprFactory->register_expression( scinew DensCalc( densityStarTag, interp->clone(), rhoEtaTags,reiEtaTags, ivarNames ) );
        gh.rootIDs.insert(densStarID);
        const Expr::ExpressionID dens2StarID = gh.exprFactory->register_expression( scinew DensCalc( density2StarTag, interp->clone(), rhoEtaTags, reiEtaTags, ivarNames ) );
        gh.rootIDs.insert(dens2StarID);
      }
      //============================================================
      // Note that it is currently assumed that we solve transport 
      // equation for non density weighted independent variables as
      // well, so, we have them in SolnVarTags/SolnVarStarTags and we 
      // should be able to update them for the next RK time stage
      //============================================================
      
      // This calculations should only take place for variable density cases when we have both momentum and scalar transport equations
      //___________________________________
      // first we estimate the density field at RK stage "*"
      if (doDenstPlus && cat==ADVANCE_SOLUTION) {    
        // Here the soln variables in density weighted form will be separated to generate rhoEta at the "*" stage
        const TagNames& tagNames = TagNames::self();
        bool found;
        for ( Expr::TagList::const_iterator i=rhoEtaTags.begin(); i!=rhoEtaTags.end(); ++i ){
          const Expr::Tag rhoEtaStarTag = Expr::Tag(i->name() + tagNames.star, Expr::STATE_NONE);
          rhoEtaStarTags.push_back(rhoEtaStarTag);                      
        }
        for ( Expr::TagList::const_iterator i=ivarNames.begin(); i!=ivarNames.end(); ++i ){
          const Expr::Tag ivarStarTag = Expr::Tag(i->name() + tagNames.star, Expr::STATE_NONE);
          ivarStarTags.push_back(ivarStarTag);                      
        }
        for ( Expr::TagList::const_iterator i=reiEtaTags.begin(); i!=reiEtaTags.end(); ++i ){
          const Expr::Tag reiEtaStarTag = Expr::Tag(i->name() + tagNames.star, Expr::STATE_NONE);
          reiEtaStarTags.push_back(reiEtaStarTag);                      
        }
        
        /* for( Expr::TagList::const_iterator i=solnVarStarTags.begin(); i!=solnVarStarTags.end(); ++i ){
          found = false;
          for( Expr::TagList::const_iterator j=ivarNames.begin(); j!=ivarNames.end(); ++j )
            if (i->name().substr(0,i->name().length()- tagNames.star.length()) == j->name()) {
              found = true;
              break;
            }
          if (!found)
            rhoEtaStarTags.push_back(*i);            
        }
        // Here the solnvariables in non density weighted form will be separated to form independant variables at RK stage with the other ReIetas at this stage
        for( Expr::TagList::const_iterator j=ivarNames.begin(); j!=ivarNames.end(); ++j ) {
          ivarStarTags.push_back( Expr::Tag(j->name() + tagNames.star, Expr::STATE_NONE) );
          found = false;
          for( Expr::TagList::const_iterator i=solnVarStarTags.begin(); i!=solnVarStarTags.end(); ++i )
            if (i->name().substr(0,i->name().length()- tagNames.star.length()) == j->name()) {
              found = true;
              break;
            }
          if (!found){
            Expr::Tag reiEtaStar = Expr::Tag(j->name() + tagNames.star, Expr::STATE_NONE);
            reiEtaStarTags.push_back(reiEtaStar);
          }
        }
         */
        // register the expression for density at RK time stage
        densityStarTag = Expr::Tag(densityTag.name() + tagNames.star, Expr::CARRY_FORWARD);
        const Expr::ExpressionID densStar = gh.exprFactory->register_expression( scinew DensCalc( densityStarTag, interp->clone(), rhoEtaStarTags, reiEtaStarTags, ivarStarTags ) );
        gh.exprFactory->cleave_from_children ( densStar );

        //___________________________________
        // and now we can estimate the density value at the next RK stage, which is "**"
        
        // Here the soln variables in density weighted form will be separated to generate rhoEta at the "**" stage
        for ( Expr::TagList::const_iterator i=rhoEtaTags.begin(); i!=rhoEtaTags.end(); ++i ){
          const Expr::Tag rhoEta2StarTag = Expr::Tag(i->name() + tagNames.doubleStar, Expr::STATE_NONE);
          rhoEta2StarTags.push_back(rhoEta2StarTag);                      
        }
        for ( Expr::TagList::const_iterator i=ivarNames.begin(); i!=ivarNames.end(); ++i ){
          const Expr::Tag ivar2StarTag = Expr::Tag(i->name() + tagNames.doubleStar, Expr::STATE_NONE);
          ivar2StarTags.push_back(ivar2StarTag);                      
        }
        for ( Expr::TagList::const_iterator i=reiEtaTags.begin(); i!=reiEtaTags.end(); ++i ){
          const Expr::Tag reiEta2StarTag = Expr::Tag(i->name() + tagNames.doubleStar, Expr::STATE_NONE);
          reiEta2StarTags.push_back(reiEta2StarTag);                      
        }
        
        /* for( Expr::TagList::const_iterator i=solnVar2StarTags.begin(); i!=solnVar2StarTags.end(); ++i ){
          found = false;
          for( Expr::TagList::const_iterator j=ivarNames.begin(); j!=ivarNames.end(); ++j )
            if (i->name().substr(0,i->name().length()-tagNames.doubleStar.length()) == j->name()) {
              found = true;
              break;
            }
          if (!found)
            rhoEta2StarTags.push_back(*i);            
        }
        // Here the solnvariables in non density weighted form will be separated to form independant variables at RK stage with the other ReIetas at this stage
        for( Expr::TagList::const_iterator j=ivarNames.begin(); j!=ivarNames.end(); ++j ) {
          ivar2StarTags.push_back( Expr::Tag(j->name() + tagNames.doubleStar, Expr::STATE_NONE) );
          found = false;
          for( Expr::TagList::const_iterator i=solnVar2StarTags.begin(); i!=solnVar2StarTags.end(); ++i )
            if (i->name().substr(0,i->name().length()-tagNames.doubleStar.length()) == j->name()) {
              found = true;
              break;
            }
          if (!found){
            Expr::Tag reiEta2Star = Expr::Tag(j->name() + tagNames.doubleStar, Expr::STATE_NONE);
            reiEta2StarTags.push_back(reiEta2Star);
          }
        }
         */        
        // register the expression for density at RK time stage
        density2StarTag = Expr::Tag(densityTag.name() + tagNames.doubleStar, Expr::CARRY_FORWARD);
        const Expr::ExpressionID dens2Star = gh.exprFactory->register_expression( scinew DensCalc( density2StarTag, interp->clone(), rhoEta2StarTags, reiEta2StarTags, ivar2StarTags ) );
        gh.exprFactory->cleave_from_children ( dens2Star );
        
      }

    }

  }
  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& params,
                             GraphCategories& gc )
  {
    //_________________________________________________________________________________
    // extracting the density tag in the cases that it is needed and also throwing the
    // error messages in different error conditions regarding to the input file

    Uintah::ProblemSpecP densityParams  = params->findBlock("Density");
    Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");

    if (tabPropsParams) {
      if (tabPropsParams->findBlock("ExtractDensity") && !densityParams) {
        std::ostringstream msg;
        msg << "ERROR: You need a tag for density when you want to extract it using TabProps." << endl
            << "       Please include the \"Density\" block in wasatch in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    for( Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
         tabPropsParams != 0;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") ){

      // determine which task list this goes on
      Category cat = parse_tasklist( tabPropsParams,false);

      // Check to see if we have scalar transport equation already set up and the problem is in variable denstiy, so we can obtain solution variables in order to estimate their values at "*" RK stage to be able to estimate the value of density at this RK stage
      Uintah::ProblemSpecP transEqnParams  = params->findBlock("TransportEquation");      
      Uintah::ProblemSpecP momEqnParams  = params->findBlock("MomentumEquations");      
      bool isConstDensity;
      densityParams->get("IsConstant",isConstDensity);

      Expr::TagList solnVarStarTags=Expr::TagList();
      Expr::TagList solnVar2StarTags=Expr::TagList();
      const bool doDenstPlus = momEqnParams && transEqnParams && !isConstDensity;
      if (doDenstPlus && cat==ADVANCE_SOLUTION) {
        std::string solnVarName;
        
        Expr::Tag solnVarTag, solnVarRHSTag, solnVarStarTag;
        Expr::Tag solnVarRHSStarTag, solnVar2StarTag;
        const TagNames& tagNames = TagNames::self();
        
        for( ; transEqnParams != 0; transEqnParams=transEqnParams->findNextBlock("TransportEquation") ) {
          transEqnParams->get( "SolutionVariable", solnVarName );
          
          // Here we get the variables needed for calculations at the stage "*" 
          solnVarTag = Expr::Tag( solnVarName, Expr::STATE_N );
          solnVarRHSTag = Expr::Tag( solnVarName+"_rhs", Expr::STATE_NONE);
          solnVarStarTag =  Expr::Tag( solnVarName+tagNames.star, Expr::STATE_NONE) ;

          if( !gc[ADVANCE_SOLUTION]->exprFactory->have_entry( solnVarStarTag ) ){
            gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew SolnVarEst<SVolField>::Builder( solnVarStarTag, solnVarTag, solnVarRHSTag, tagNames.timestep ));
          }
          solnVarStarTags.push_back( solnVarStarTag );          
          
          // Here we get the variables needed for calculations at the stage "**" 
          solnVarRHSStarTag = Expr::Tag( solnVarName+"_rhs"+tagNames.star, Expr::STATE_NONE);
          solnVar2StarTag =  Expr::Tag( solnVarName+tagNames.doubleStar, Expr::STATE_NONE) ;
          
          if( !gc[ADVANCE_SOLUTION]->exprFactory->have_entry( solnVar2StarTag ) ){
            gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew SolnVarEst<SVolField>::Builder( solnVar2StarTag, solnVarStarTag, solnVarRHSStarTag, tagNames.timestep ));
          }
          solnVar2StarTags.push_back( solnVar2StarTag );          
          
        }
      }
      parse_tabprops( tabPropsParams, *gc[cat], cat, doDenstPlus );
    }
  }

  //====================================================================

} // namespace Wasatch
