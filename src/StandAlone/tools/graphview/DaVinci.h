/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DAVINCI_H
#define DAVINCI_H

#include <fstream>
#include <vector>
#include <list>
#include <unistd.h>

class GV_TaskGraph;
class GV_Task;
class Edge;

class DaVinci {
public:
  enum Orientation {TOP_DOWN, BOTTOM_UP, LEFT_RIGHT, RIGHT_LEFT};
#if 0
  enum {
    EVT_DV_QUIT = 'DVQT',
    EVT_DV_SELECT_NODE = 'DVSN',
    EVT_DV_DOUBLE_CLICK_NODE = 'DVDN',
    EVT_DV_SELECT_EDGE = 'DVSE',
    EVT_DV_DOUBLE_CLICK_EDGE = 'DVDE'       
  };
#endif
  enum {EVT_DV_QUIT, 
        EVT_DV_SELECT_NODE, 
        EVT_DV_DOUBLE_CLICK_NODE,
        EVT_DV_SELECT_EDGE,
        EVT_DV_DOUBLE_CLICK_EDGE
  };

  static DaVinci* run();

  ~DaVinci();

  void handleInput();

  void setGraph(const GV_TaskGraph* graph);
  void setOrientation(Orientation orientation);
  void setFontSize(int font_size);

  int getOutput() const { return m_fromDV; }

  const std::string& getSelectedEdge() const
  { return m_selectedEdge; }

  const std::list<std::string>& getSelectedNodes() const
  { return m_selectedNodes; }

  static bool doExclusion;
private:
  DaVinci(pid_t pid, int to, int from);

  // Parses a DaVinci answer string simply by breaking it
  // the cmd and arguments (by inserting '\0's in the cmd
  // string and appending char*'s the the args list.
  void parseAnswer(char* cmd, std::list<char*>& args);

  pid_t m_PID;
  int m_toDV;
  int m_fromDV;
  std::list<std::string> m_selectedNodes;
  std::string m_selectedEdge;

  // prevent copying and assignment
  DaVinci(const DaVinci& rhs);
  DaVinci& operator=(const DaVinci& rhs);
};

#endif // DAVINCI_H



