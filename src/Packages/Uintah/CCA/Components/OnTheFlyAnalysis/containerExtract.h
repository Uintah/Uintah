#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_containerExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_containerExtract_h
#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {


  /**************************************

    CLASS
    containerExtract

    GENERAL INFORMATION

    containerExtract.h

    Steve Brown
    Todd Harman
    Department of Mechanical Engineering
    University of Utah

   ****************************************/
  class containerExtract : public AnalysisModule {
    public:
      containerExtract(ProblemSpecP& prob_spec,
          SimulationStateP& sharedState,
          Output* dataArchiver);
      containerExtract();

      virtual ~containerExtract();

      virtual void problemSetup(const ProblemSpecP& prob_spec,
          GridP& grid,
          SimulationStateP& sharedState);

      virtual void scheduleInitialize(SchedulerP& sched,
          const LevelP& level);

      virtual void restartInitialize();

      virtual void scheduleDoAnalysis(SchedulerP& sched,
          const LevelP& level);


    private:
      // general labels
      class containerExtractLabel {
        public:
          VarLabel* lastWriteTimeLabel;
      };
      containerExtractLabel* ps_lb;

      enum EXTRACT_MODE { INTERIOR, SURFACE, INCIDENT, NET, VELOCITY };
      enum FACE { NONE, TOP, BOTTOM, NORTH, SOUTH, EAST, WEST };

      /* Inner class extractCell.
          The main driving structure at extract-time.
          You may specify one grid cell and one VarLabel per extractCell.
          (To do a multiplicity of labels/cells just duplicate them.)

          This would easily convert to a generic 'point locus' format
          to add an abstraction layer to other AnalysisModule types.

          I.e. lineExtract pointExtract containerExtract would all
          assemble a locus of extractCells and then pass them on to
          the abstract extract-time driver.
          */
      class extractCell {    
        public:
          enum EXTRACT_MODE type;
          enum FACE face; //only used for incident net velocity
          IntVector c;
          VarLabel* vl;

          extractCell() { type = INCIDENT; face = TOP; c = IntVector(-1,-1,-1); vl = NULL; }
          extractCell(enum EXTRACT_MODE t, enum FACE f, IntVector i, VarLabel* v):
            type(t), face(f), c(i), vl(v) {}
      };

      struct extractVarLabel {
        enum EXTRACT_MODE mode;
        VarLabel* vl;
      };

      struct container {
        string name;  
        vector<extractVarLabel*> vls;
        vector<GeometryPieceP> geomObjs;
        vector<IntVector> containerPoints;
        vector<extractCell*> extractCells; 
      };

      UINTAHSHARE friend std::ostream& operator<<(std::ostream& ostr, const extractCell& exc) {
        return ostr << *(exc.vl) << "_" << exc.c.x() << "_" << exc.c.y() << "_" << exc.c.z() << ".dat";
      }

      //__________________________________
      // global constants
      double d_writeFreq; 
      double d_StartTime;
      double d_StopTime;
      vector<VarLabel*> d_varLabels;
      SimulationStateP d_sharedState;
      vector<container*> d_containers;
      Output* d_dataArchiver;
      ProblemSpecP d_prob_spec;
      const Material* d_matl;
      MaterialSet* d_matl_set;

      void initialize(const ProcessorGroup*, 
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse*,
          DataWarehouse* new_dw);

      void doAnalysis(const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset*,
          DataWarehouse*,
          DataWarehouse* new_dw);

      void createFile(string& filename, extractCell& e);

      void createDirectory(string& lineName, string& levelIndex);




  };

}

#endif
