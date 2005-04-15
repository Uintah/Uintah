#include <list>
#include <map>
#include <string>

class DOM_Document;

class Edge;
class TaskGraph;

class Task {
  friend class Edge;
  friend class TaskGraph;
  
private:
  Task(std::string name, double duration, TaskGraph* owner);

public:
  ~Task();
  
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

  TaskGraph* getGraph() const
  { return m_graph; }
  
  Edge* addDependency(Task* task);

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
  TaskGraph* m_graph;
  std::list<Edge*> m_dependencyEdges;
  std::list<Edge*> m_dependentEdges;

  // maximum cost for any path below this task (paths of dependents)
  double m_maxBelowCost;
  
  // maximum cost for any path above this task (paths of dependencies)
  double m_maxAboveCost;
};

class Edge {
  friend class Task;

private:
  Edge(Task* source, Task* target);    

public:
  Task* getSource() const { return m_source; }
  Task* getTarget() const { return m_target; }

  TaskGraph* getGraph() const
  { return m_source->getGraph(); }

  double getMaxInclusivePathCost() const
  { return m_source->getMaxInclBelowCost() + m_target->getMaxInclAboveCost(); }

  float getMaxPathPercent() const;
  
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
  Task* m_source; // dependent
  Task* m_target; // dependency
};

class TaskGraph {
public:
  static TaskGraph* inflate(std::string xmlDir);
  
  ~TaskGraph();
  
  Task* findTask(std::string name);
  Edge* findEdge(std::string name);
  const std::list<Task*> getTasks() const { return m_tasks; }
  const std::list<Edge*> getEdges() const { return m_edges; }

  // Set the threshold percent for hiding tasks whose
  // maximum inclusive paths are under the given threshold.
  void setThresholdPercent(float percent)
  { m_thresholdPercent = percent; }
  
  float getThresholdPercent() const
  { return m_thresholdPercent; }
  
  double getCriticalPathCost() const
  { return m_criticalPathCost; }
private:
  // Creates a graph from an array of documents.
  // Each document contains node information for a sub-set
  // of the graph, but they should all contain the same edges.
  TaskGraph(std::list<DOM_Document> xmlDoc);

  // Compute the maximum paths above and below each
  // node as well as the critical path cost.
  void computeMaxPathLengths(); // called when graph is created

  std::list<Task*> m_tasks;
  std::list<Edge*> m_edges;
  std::map<std::string, Task*> m_taskMap;
  std::map<std::string, Edge*> m_edgeMap;
  double m_criticalPathCost;
  float m_thresholdPercent;
};
