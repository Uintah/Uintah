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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/NullDamage.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/JohnsonCookDamage.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/HancockMacKenzieDamage.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ThresholdDamage.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/BrittleDamage.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
//#include <iostream>
#include <string>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;
//______________________________________________________________________
//
DamageModel* DamageModelFactory::create(ProblemSpecP    & matl_ps,
                                        MPMFlags        * flags,
                                        MaterialManager * materialManager)
{   
  ProblemSpecP child = matl_ps->findBlock("damage_model");
  if(!child) {
    proc0cout << "**WARNING** Creating default null damage model" << endl;
    return( scinew NullDamage() );
  }
  
  string dam_type = "none";
  if(!child->getAttribute( "type", dam_type )){
    throw ProblemSetupException("No type for damage_model", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  bulletproofing
  string cm_type = "none";

  if ( matl_ps->getNodeName() != "constitutive_model" ) { 
    ProblemSpecP cm_ps = matl_ps->findBlock("constitutive_model");
    cm_ps->getAttribute("type", cm_type);
  }
  
  if (dam_type == "hancock_mackenzie" || dam_type == "johnson_cook" ) {
    if( cm_type != "elastic_plastic_hp" && cm_type != "elastic_plastic"  ){
      string txt="MPM:  The only constitute model that works with the johnson_cook and hancock_mackenzie damage models is elastic_plastic/elastic_plastic_hp";
      txt = txt + " (CM: " + cm_type + " DM: " + dam_type + ")"; 
      throw ProblemSetupException(txt, __FILE__, __LINE__);
    }
  }  
  
  
  //__________________________________
  //
  if (dam_type == "johnson_cook"){
    return( scinew JohnsonCookDamage( child ) );
  }
  else if (dam_type == "hancock_mackenzie") {
    return( scinew HancockMacKenzieDamage( child ) );
  }
  else if (dam_type == "Threshold") {
    return( scinew ThresholdDamage( child, flags, materialManager ) );
  }
  else if (dam_type == "Brittle") {
    return( scinew BrittleDamage( child ) );
  }
  else if (dam_type == "none") {
    return(scinew NullDamage( child ) );
  }
  else {
    proc0cout << "**WARNING** Creating default null damage model" << endl;
    return( scinew NullDamage( child ) );
    //throw ProblemSetupException("Unknown Damage Model ("+dam_type+")", __FILE__, __LINE__);
  }
}
//______________________________________________________________________
//
DamageModel* DamageModelFactory::createCopy(const DamageModel* dm)
{
  if (dynamic_cast<const JohnsonCookDamage*>(dm)){
    return(scinew JohnsonCookDamage(dynamic_cast<const JohnsonCookDamage*>(dm)));
  }
  else if (dynamic_cast<const HancockMacKenzieDamage*>(dm)) {
    return(scinew HancockMacKenzieDamage(dynamic_cast<const HancockMacKenzieDamage*>(dm)));
 }
  else {
    proc0cout << "**WARNING** Creating copy of default null damage model" << endl;
    return(scinew NullDamage(dynamic_cast<const NullDamage*>(dm)));
    //throw ProblemSetupException("Cannot create copy of unknown damage model", __FILE__, __LINE__);
  }
}
