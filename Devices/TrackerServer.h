

#ifndef sci_Devices_TrackerServer_h
#define sci_Devices_TrackerServer_h 1

struct TrackerPosition {
    int x, y, z;
    short pitch, yaw, roll;
    unsigned int out:1;
    unsigned int fringe:1;
    unsigned int s:1;
    unsigned int l:1;
    unsigned int m:1;
    unsigned int r:1;
    int operator!=(const TrackerPosition&);
};

struct TrackerData {
    int mouse_moved;
    TrackerPosition mouse_pos;
    int head_moved;
    TrackerPosition head_pos;
};

int GetTrackerData(TrackerData&);

#endif
