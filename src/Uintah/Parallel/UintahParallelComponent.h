
#ifndef Uintah_Parallel_UintahParallelComponent_h
#define Uintah_Parallel_UintahParallelComponent_h

#include <string>
class UintahParallelPort;

class UintahParallelComponent {
public:
    UintahParallelComponent();
    virtual ~UintahParallelComponent();

    void setPort(const std::string& name, UintahParallelPort* port);
};

#endif
