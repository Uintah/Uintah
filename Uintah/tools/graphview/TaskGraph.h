#include <list>
#include <map>
#include <string>

class DOM_Document;

class Edge;

class Task {
public:
    Task(std::string name, double duration);
    ~Task();
    
    std::string getName() const { return m_name; }
    double getDuration() const { return m_duration; }
    const std::list<Task*>& getDependencies() const { return m_dependencies; }
    const std::list<Task*>& getDependents() const { return m_dependents; }
    Edge* addDependency(Task* task);

private:
    std::string m_name;
    double m_duration;
    std::list<Task*> m_dependencies;
    std::list<Task*> m_dependents;
};

class Edge {
    friend class Task;

private:
    Edge(Task* source, Task* target);    

public:
    Task* getSource() { return m_source; }
    Task* getTarget() { return m_target; }
    
private:
    Task* m_source; // dependent
    Task* m_target; // dependency
};

class TaskGraph {
public:
    static TaskGraph* inflate(std::string xmlFileName);
    
    ~TaskGraph();
    
    Task* find(std::string name);
    const std::list<Task*> getTasks() const { return m_tasks; }

private:
    TaskGraph(DOM_Document xmlDoc);

    std::list<Task*> m_tasks;
    std::list<Edge*> m_edges;
    std::map<std::string, Task*> m_taskMap;
};
