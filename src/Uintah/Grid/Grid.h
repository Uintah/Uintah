
#ifndef UINTAH_HOMEBREW_GRID_H
#define UINTAH_HOMEBREW_GRID_H

#include "GridP.h"
#include "Handle.h"
#include "LevelP.h"
#include "RefCounted.h"
#include <vector>

class Grid : public RefCounted {
public:
    Grid();
    virtual ~Grid();

    int numLevels() const;
    LevelP& getLevel(int idx);
    void addLevel(const LevelP& level);
private:
    std::vector<LevelP> levels;

    Grid(const Grid&);
    Grid& operator=(const Grid&);
};

#endif
