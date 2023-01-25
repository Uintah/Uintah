/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#ifndef UINTAH_HOMEBREW_CZLABEL_H
#define UINTAH_HOMEBREW_CZLABEL_H

#include <vector>

namespace Uintah {

  class VarLabel;

    class CZLabel {
    public:

      CZLabel();
      ~CZLabel();

      // For Cohesive Zones
      const VarLabel* czAreaLabel; 
      const VarLabel* czAreaLabel_preReloc; 
      const VarLabel* czNormLabel; 
      const VarLabel* czNormLabel_preReloc; 
      const VarLabel* czTangLabel; 
      const VarLabel* czTangLabel_preReloc; 
      const VarLabel* czDispTopLabel; 
      const VarLabel* czDispTopLabel_preReloc; 
      const VarLabel* czDispBottomLabel; 
      const VarLabel* czDispBottomLabel_preReloc; 
      const VarLabel* czSeparationLabel; 
      const VarLabel* czSeparationLabel_preReloc; 
      const VarLabel* czForceLabel; 
      const VarLabel* czForceLabel_preReloc; 
      const VarLabel* czTopMatLabel; 
      const VarLabel* czTopMatLabel_preReloc; 
      const VarLabel* czBotMatLabel; 
      const VarLabel* czBotMatLabel_preReloc; 
      const VarLabel* czFailedLabel; 
      const VarLabel* czFailedLabel_preReloc; 
      const VarLabel* czIDLabel; 
      const VarLabel* czIDLabel_preReloc; 
      const VarLabel* pCellNACZIDLabel;
    };
} // End namespace Uintah

#endif
