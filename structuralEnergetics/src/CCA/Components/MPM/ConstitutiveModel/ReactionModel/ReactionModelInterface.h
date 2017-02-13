/*
 * ReactionModelInterface.h
 *
 *  Created on: Feb 9, 2017
 *      Author: jbhooper
 *
 *
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef SRC_CCA_COMPONENTS_MPM_CONSTITUTIVEMODEL_REACTIONMODEL_REACTIONMODELINTERFACE_H_
#define SRC_CCA_COMPONENTS_MPM_CONSTITUTIVEMODEL_REACTIONMODEL_REACTIONMODELINTERFACE_H_

namespace Uintah
{
  class ReactionModel
  {
    public:
               ReactionModel();
      virtual ~ReactionModel();

      VarLabel* getReactionProgressLabel()
      {
        return reactionProgressLabel;
      }
      VarLabel* getReactionInterfaceLabel()
      {
        return reactionInterfaceLabel;
      }
      int       getOtherReactantDWI()
      {
        return otherReactantDWI;
      }

    private:
      VarLabel* reactionProgressLabel;
      VarLabel* reactionInterfaceLabel;
      double    d_dH_Rxn;
      bool      d_continuousReaction; // Add incremental dH or wait until entire MP is reacted?
      int       otherReactantDWI;
      // Holds the varlabels necessary to calculate reaction here.
      std::vector<VarLabel*>  otherReactantLabels;
  };
}



#endif /* SRC_CCA_COMPONENTS_MPM_CONSTITUTIVEMODEL_REACTIONMODEL_REACTIONMODELINTERFACE_H_ */
