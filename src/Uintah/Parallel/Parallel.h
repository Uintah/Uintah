
#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

class Parallel {
public:
    static void initializeManager(int argc, char** argv);
    static void finalizeManager();
private:
    Parallel();
    Parallel(const Parallel&);
    ~Parallel();
    Parallel& operator=(const Parallel&);
};

#endif
