#ifndef DAVINCI_H
#define DAVINCI_H

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>
#include <unistd.h>

class GV_TaskGraph;
class GV_Task;
class Edge;

class DaVinci {
public:
  enum Orientation {TOP_DOWN, BOTTOM_UP, LEFT_RIGHT, RIGHT_LEFT};
  enum {
    EVT_DV_QUIT = 'DVQT',
    EVT_DV_SELECT_NODE = 'DVSN',
    EVT_DV_DOUBLE_CLICK_NODE = 'DVDN',
    EVT_DV_SELECT_EDGE = 'DVSE',
    EVT_DV_DOUBLE_CLICK_EDGE = 'DVDE'       
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



