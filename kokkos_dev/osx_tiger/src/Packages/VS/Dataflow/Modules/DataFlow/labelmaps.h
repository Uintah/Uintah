/******************************************************************************
 * File: labelmaps.h
 *
 * Description: C header source class definitions to provide an API for
 *		the Visible Human KnowledgeBase
 *
 *              Included in the KnowledgeBase are:
 *
 *              o segmentation information as formatted ASCII files:
 *		  * The Master Anatomy Label Map
 *		  * Spatial Adjacency relations for the anatomical structures
 *		  and
 *		  * Bounding Boxes for each anatomical entity.
 *
 *              o An Injury list in XML format
 *
 *              o A mapping from anatomical structures to physiological
 *                parameters corresponding to that structure or region
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#ifndef LABELMAPS_H
#define LABELMAPS_H

#include <string>
#include <vector>

using namespace std;

#define VH_LM_NUM_NAMES 2500

/* misc string manip functions */
char *space_to_underbar(char *dst, char *src);

char *capitalize(char *dst, char *src);

/******************************************************************************
 * class VH_MasterAnatomy
 *
 * description: Two parallel arrays -- names <-> indices
 *
 *		The map from anatomical volume (e.g., C-T, MRI) tags to
 *		anatomical names.
 ******************************************************************************/

class VH_MasterAnatomy {
public:
	VH_MasterAnatomy();
	~VH_MasterAnatomy();
	void readFile(char *infilename);
	void readFile(FILE *infileptr);
	string getActiveFile() { return activeFile; };
	char *get_anatomyname(int labelindex);
	int get_labelindex(char *anatomyname);
	int get_num_names() { return num_names; };
	int get_max_labelindex() { return maxLabelIndex; };
private:
	string activeFile;
	char **anatomyname;
	int *labelindex;
	int num_names;
	int maxLabelIndex;
};

/******************************************************************************
 * class VH_AdjacencyMapping
 *
 * description: An array of integers for each anatomical entity -- the
 *		relation describes every entity spatially adjacent to the
 *		indexed entity.
 ******************************************************************************/

#define VH_FILE_MAXLINE 2048

class VH_AdjacencyMapping {
public:
	VH_AdjacencyMapping();
	~VH_AdjacencyMapping();
	void readFile(char *infilename);
        void readFile(FILE *infileptr);
	string getActiveFile() { return activeFile; };
	int get_num_rel(int index);
	int get_num_names() { return num_names; };
	int *adjacent_to(int index);
private:
	string activeFile;
	int **rellist;
	int *numrel;
	int num_names;
};

/******************************************************************************
 * class VH_AnatomyBoundingBox
 *
 * description: A doubly-linked list of nodes consisting of an ASCII
 *		char *name -- matching a tissue in the MasterAnatomy
 *              and the X-Y-Z extrema of the segmentation of that
 *              tissue.  Note: dimensions are integer Voxel addresses
 *		referring to the original segmented volume.
 ******************************************************************************/
class VH_AnatomyBoundingBox {
private:
	char *anatomyname_;
// Modification on April 28, 2005 by R. C. Ward
// Bounding box coorindates are now floating point
        float minX_, maxX_, minY_, maxY_, minZ_, maxZ_, minSlice_, maxSlice_;
//	int minX_, maxX_, minY_, maxY_, minZ_, maxZ_, minSlice_, maxSlice_;
	VH_AnatomyBoundingBox *blink, *flink;
public:
        VH_AnatomyBoundingBox() { flink = blink = this; };
        VH_AnatomyBoundingBox *next() { return flink; };
        VH_AnatomyBoundingBox *prev() { return blink; };
        void set_next(VH_AnatomyBoundingBox *new_next) { flink = new_next; };
        void set_prev(VH_AnatomyBoundingBox *new_prev) { blink = new_prev; };
	void append(VH_AnatomyBoundingBox *newNode);
        void readFile(FILE *infileptr);
	char *get_anatomyname() { return(anatomyname_); };
	void set_anatomyname(char *newName) { anatomyname_ = newName; };
// Modification April 28, 2005 by R. C. Ward
// Bounding Box min, max are now doubles
        void set_minX(float new_minX) { minX_ = new_minX; };
        void set_maxX(float new_maxX) { maxX_ = new_maxX; };
        void set_minY(float new_minY) { minY_ = new_minY; };
        void set_maxY(float new_maxY) { maxY_ = new_maxY; };
        void set_minZ(float new_minZ) { minZ_ = new_minZ; };
        void set_maxZ(float new_maxZ) { maxZ_ = new_maxZ; };
        void set_minSlice(float newMinSlice) { minSlice_ = newMinSlice; };
        void set_maxSlice(float newMaxSlice) { maxSlice_ = newMaxSlice; };
//        void set_minX(int new_minX) { minX_ = new_minX; };
//        void set_maxX(int new_maxX) { maxX_ = new_maxX; };
//        void set_minY(int new_minY) { minY_ = new_minY; };
//        void set_maxY(int new_maxY) { maxY_ = new_maxY; };
//        void set_minZ(int new_minZ) { minZ_ = new_minZ; };
//        void set_maxZ(int new_maxZ) { maxZ_ = new_maxZ; };
//        void set_minSlice(int newMinSlice) { minSlice_ = newMinSlice; };
//        void set_maxSlice(int newMaxSlice) { maxSlice_ = newMaxSlice; };
        float get_minX() { return minX_; };
        float get_maxX() { return maxX_; };
        float get_minY() { return minY_; };
        float get_maxY() { return maxY_; };
        float get_minZ() { return minZ_; };
        float get_maxZ() { return maxZ_; };
        float get_minSlice() { return minSlice_; };
        float get_maxSlice() { return maxSlice_; };
//        int get_minX() { return minX_; };
//        int get_maxX() { return maxX_; };
//        int get_minY() { return minY_; };
//        int get_maxY() { return maxY_; };
//        int get_minZ() { return minZ_; };
//        int get_maxZ() { return maxZ_; };
//        int get_minSlice() { return minSlice_; };
//        int get_maxSlice() { return maxSlice_; };

};

void
VH_Anatomy_deleteBBox_node(VH_AnatomyBoundingBox *delNode);

void
VH_Anatomy_destroyBBox_list(VH_AnatomyBoundingBox *delList);

/******************************************************************************
 * Read an ASCII AnatomyBoundingBox file into a linked list
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_readBoundingBox_File(char *infilename);

/******************************************************************************
 * Find the boundingBox of a named anatomical entity
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findBoundingBox(VH_AnatomyBoundingBox *list, char *anatomyname);

/******************************************************************************
 * Find the largest bounding volume of the segmentation
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findMaxBoundingBox(VH_AnatomyBoundingBox *list);

/******************************************************************************
 * class VH_injury
 *
 * description: A node consisting of the name of an injured tissue, timestamp
 *              when this tissue becaem injured  and iconic geometry to display
 *              to indicate the extent of the injury.
 ******************************************************************************/

#define UNSET -1
#define SET_AXIS_START_POINT 0
#define SET_AXIS_END_POINT   1
#define SET_DIAMETER         2
#define SET_INSIDE_DIAMETER  3
#define SET_LENGTH           4 

class VH_injury
{
  public:
  string anatomyname;
  int timeStamp;
  // flags to test our tree traversal context
  int context;
  bool isPrimaryInjury, isSecondaryInjury, isGeometry;
  bool isAblate, isStun;
  // flags to test whether this node is complete
  bool timeSet, nameSet, point0set, point1set, rad0set, rad1set;
  bool inside_rad0set, inside_rad1set;
  string geom_type; // line, sphere, cylinder, hollow cylinder
  float axisX0, axisY0, axisZ0; // center axis endpoint 0
  float axisX1, axisY1, axisZ1; // center axis endpoint 1
  float inside_rad0, inside_rad1, rad0, rad1, len;
  float probability;

  VH_injury() { context = UNSET; rad0set = rad1set = false;
                inside_rad0set = inside_rad1set = false;
                nameSet = timeSet = point0set = point1set = false;
		isPrimaryInjury = isSecondaryInjury = isGeometry = false;
		isAblate = isStun = false; probability = 0.0;
                rad0 = rad1 = 0.0; };
  VH_injury(char *newName) { anatomyname = string(newName);
             context = UNSET; rad0set = rad1set = false;
             inside_rad0set = inside_rad1set = false;
             nameSet = true; timeSet = point0set = point1set = false;
             isPrimaryInjury = isSecondaryInjury = isGeometry = false;
             isAblate = isStun = false; probability = 0.0;
             rad0 = rad1 = 0.0; };
  bool iscomplete();
  void print();
};

bool
is_injured(char *targetName, vector<VH_injury> &injured_tissue_list);

/******************************************************************************
 * class VH_HIPvarMap
 *
 * description: Two parallel arrays -- FMA names <-> HIP var file names
 *
 *		The map from anatomical anatomical names to HIP data channels.
 ******************************************************************************/

class VH_HIPvarMap {
public:
	VH_HIPvarMap();
	~VH_HIPvarMap();
	void readFile(char *infilename);
	void readFile(FILE *infileptr);
	string getActiveFile() { return activeFile; };
	char *get_HIPvarFile(char *targAnatomyName);
	int get_num_names() { return num_names; };
private:
	string activeFile;
	char **anatomyname;
	char **HIPvarFileName;
	int num_names;
};

/******************************************************************************
 * class VH_physioMapping
 *
 * description: A node containing a mapping from anatomical names to
 *		sets of corresponding physiological parameters
 ******************************************************************************/

class VH_hipParam
{
  public:
        VH_hipParam();
        VH_hipParam(int, char *, char *, char *, char *);
        int col_no;
        string var_shortName;
        string var_longName;
        string var_type;
        string var_unit;
};

class VH_physioMapping
{
  private:
        string anatomyLump;
        string GE_anatName;
        string eFMA_prefName;
	int fmaID;
  public:
        vector<VH_hipParam *> hip_param;
        VH_physioMapping();
	VH_physioMapping(char *, char *,  char *, int);
};

VH_physioMapping *VH_physioMapping_readFile(char *inFileName);

#endif
