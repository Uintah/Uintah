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

#ifndef UINTAH_HOMEBREW_SHELLMPMLABEL_H
#define UINTAH_HOMEBREW_SHELLMPMLABEL_H

#include <vector>

namespace Uintah {

  class VarLabel;

    class ShellMPMLabel {
    public:

      ShellMPMLabel();
      ~ShellMPMLabel();

      const VarLabel* pThickTopLabel;
      const VarLabel* pInitialThickTopLabel;
      const VarLabel* pThickBotLabel;
      const VarLabel* pInitialThickBotLabel;
      const VarLabel* pNormalLabel;
      const VarLabel* pInitialNormalLabel;
      const VarLabel* pThickTopLabel_preReloc;
      const VarLabel* pInitialThickTopLabel_preReloc;
      const VarLabel* pThickBotLabel_preReloc;
      const VarLabel* pInitialThickBotLabel_preReloc;
      const VarLabel* pNormalLabel_preReloc;
      const VarLabel* pInitialNormalLabel_preReloc;
      const VarLabel* pTypeLabel;
      const VarLabel* pTypeLabel_preReloc;

      const VarLabel* gNormalRotRateLabel; 
      const VarLabel* gNormalRotMomentLabel; 
      const VarLabel* gNormalRotMassLabel; 
      const VarLabel* gNormalRotAccLabel; 
      
    };
} // End namespace Uintah

#endif
