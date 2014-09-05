/*
 * LucretiusParsing.cc
 *
 *  Created on: Mar 17, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusParsing.h>
#include <string>
#include <vector>

namespace lucretiusParse {



  bool constrainOOPOnNegativeConstant = true;

  oopPotentialMap::oopPotentialMap(double _energeticConstant, std::string _comment)
      : planarityConstant(_energeticConstant), comment(_comment) {
    if ((planarityConstant <= 0.0) && constrainOOPOnNegativeConstant) constrainToPlanar = true;
    if (!constrainOOPOnNegativeConstant) {
      // !JBH  Should print out a warning here that negative OOP constants aren't constraining to planar
      // !     This is a very specific and esoteric concern, so can wait.  !FIXME
    }
  }

//  std::ostream& operator<< (std::ostream& osOut, const lucretiusAtomLabel& LabelOut) {
//    osOut << LabelOut.label;
//    return osOut;
//  }
//
//  std::istream& operator>> (std::istream& isIn, lucretiusAtomLabel& LabelIn) {
//    isIn >> LabelIn.label[0] >> LabelIn.label[1] >> LabelIn.label[2];
//    return isIn;
//  }

}


