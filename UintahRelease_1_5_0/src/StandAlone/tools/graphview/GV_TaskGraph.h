/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/ProblemSpec/ProblemSpecP.h>
#include <list>
#include <vector>
#include <map>
#include <string>

class Edge;
class GV_TaskGraph;

using Uintah::ProblemSpecP;

class GV_Task {
  friend class Edge;
  friend class GV_TaskGraph;
  
private:
  GV_Task(std::string name, double duration, GV_TaskGraph* owner);

public:
  ~GV_Task();
  
  std::string getName() const { return m_name; }
  double getDuration() const { return m_duration; }

  // replaces the duration if the given duration is greater than
  // the old duration (sets to the maximum of given durations).
  void testSetDuration(double duration)
  { if (m_duration < duration) m_duration = duration; }
  
  const std::list<Edge*>& getDependencyEdges() const
  { return m_dependencyEdges; }

  const std::list<Edge*>& getDependentEdges() const
  { return m_dependentEdges; }

  GV_TaskGraph* getGraph() const
  { return m_graph; }
  
  Edge* addDependency(GV_Task* task);

  double getMaxAboveCost() const
  { return m_maxAboveCost; }

  double getMaxInclAboveCost() const
  { return m_maxAboveCost + m_duration; }

  double getMaxBelowCost() const
  { return m_maxBelowCost; }

  double getMaxInclBelowCost() const
  { return m_maxBelowCost + m_duration; }

  double getMaxInclusivePathCost() const
  { return m_maxAboveCost + m_maxBelowCost + m_duration; }

  float getMaxPathPercent() const;

  void processTaskForSorting(std::vector<GV_Task*>& sortedTasks);

  bool sorted()
  { return m_sorted; }
  
  void resetFlags()
  { m_sorted = false; m_visited = false; }
private:
  // The below are called by Edge::relaxEdgeDown() and
  // Edge::relaxEdgeUp() respectively.
  
  void testSetMaxAboveCost(double test_cost)
  { if (test_cost > m_maxAboveCost) m_maxAboveCost = test_cost; }
  
  void testSetMaxBelowCost(double test_cost)
  { if (test_cost > m_maxBelowCost) m_maxBelowCost = test_cost; }

private:
  std::string m_name;
  double m_duration;
  GV_TaskGraph* m_graph;
  std::list<Edge*> m_dependencyEdges;
  std::list<Edge*> m_dependentEdges;

  // maximum cost for any path below this GV_Task (paths of dependents)
  double m_maxBelowCost;
  
  // maximum cost for any path above this GV_Task (paths of dependencies)
  double m_maxAboveCost;

  bool m_sorted;
  bool m_visited;
};

class Edge {
  friend class GV_Task;

private:
  Edge(GV_Task* source, GV_Task* target);    

public:
  GV_Task* getSource() const { return m_source; }
  GV_Task* getTarget() const { return m_target; }

  GV_TaskGraph* getGraph() const
  { return m_source->getGraph(); }

  double getMaxInclusivePathCost() const
  { return m_source->getMaxInclAboveCost() + m_target->getMaxInclBelowCost(); }

  float getMaxPathPercent() const;

  bool isObsolete() const
  {
    return m_obsolete;
  }

  void setObsolete()
  { m_obsolete = true; }
  
  // Set the maximum below cost of the target to
  // max(target->max_below_cost, source->getMaxInclBelowCost).
  // If edges are relaxed up in reverse topological order
  // for an acyclic graph, then the true maximum below path
  // costs will be set.
  void relaxEdgeUp();

  // Set the maximum above cost of the target to
  // max(target->max_above_cost, target->getMaxInclAboveCost).
  // If edges are relaxed down in topological order
  // for an acyclic graph, then the true maximum above path
  // costs will be set.
  void relaxEdgeDown();

private:
  GV_Task* m_source; // dependent
  GV_Task* m_target; // dependency
  bool m_obsolete;
};

class GV_TaskGraph {
public:
  static GV_TaskGraph* inflate(std::string xmlDir);
  
  ~GV_TaskGraph();
  
  GV_Task* findTask(std::string name);
  Edge* findEdge(std::string name);
  const std::list<GV_Task*> getTasks() const { return m_tasks; }
  const std::list<Edge*> getEdges() const { return m_edges; }

  // Set the threshold percent for hiding GV_Tasks whose
  // maximum inclusive paths are under the given threshold.
  void setThresholdPercent(float percent)
  { m_thresholdPercent = percent; }
  
  float getThresholdPercent() const
  { return m_thresholdPercent; }
  
  double getCriticalPathCost() const
  { return m_criticalPathCost; }
private:
  // Creates an empty graph
  GV_TaskGraph();

  // read the nodes from the xml document and create them in the graph
  void readNodes(ProblemSpecP xmlDoc);

  // read the edges from the xml document and create them in the graph
  void readEdges(ProblemSpecP xmlDoc);

  // Compute the maximum paths above and below each
  // node as well as the critical path cost.
  void computeMaxPathLengths(); // called when graph is created

  void topologicallySortEdges();

  void markObsoleteEdges();
  
  std::list<GV_Task*> m_tasks;
  std::list<Edge*> m_edges;
  std::map<std::string, GV_Task*> m_taskMap;
  std::map<std::string, Edge*> m_edgeMap;
  double m_criticalPathCost;
  float m_thresholdPercent;
};
