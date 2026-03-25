/*
 * Copyright © 2025 by Geocosm LLC                                   
 */

#ifndef UINTAH_HOMEBREW_LINESEGMENTLABEL_H
#define UINTAH_HOMEBREW_LINESEGMENTLABEL_H

namespace Uintah {

  class VarLabel;

    class LineSegmentLabel {
    public:

      LineSegmentLabel();
      ~LineSegmentLabel();

      const VarLabel* lineSegmentCountLabel;
      const VarLabel* linesegIDLabel; 
      const VarLabel* linesegIDLabel_preReloc; 
      const VarLabel* lsMidToEndVectorLabel; 
      const VarLabel* lsMidToEndVectorLabel_preReloc; 
    };
} // End namespace Uintah

#endif
