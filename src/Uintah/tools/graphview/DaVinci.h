#include <fstream>
#include <vector>
#include <unistd.h>

class TaskGraph;

class DaVinci {
public:
    enum Orientation {TOP_DOWN, BOTTOM_UP, LEFT_RIGHT, RIGHT_LEFT};

    static DaVinci* run();

    ~DaVinci();

    void handleInput();

    void setGraph(const TaskGraph* graph);
    void setOrientation(Orientation orientation);

    int getOutput() const { return m_fromDV; }

private:
    DaVinci(pid_t pid, int to, int from);

    pid_t m_PID;
    int m_toDV;
    int m_fromDV;

    // prevent copying and assignment
    DaVinci(const DaVinci& rhs);
    DaVinci& operator=(const DaVinci& rhs);
};
