/**************************************************************************
// Program : MPMInputsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           MPM flags and parameters
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;

//**************************************************************************
// Class   : MPMInputsPanel
//**************************************************************************
public class MPMInputsPanel extends JPanel {

  // Static variables

  // Data
  private String d_integrator = null;
  private int d_mpmAlgo = 0;
  private String d_failAlgo = null;
  private boolean d_impDynamic = true;
  private String d_solver = null;

  private boolean d_gridReset = true;
  private boolean d_accStrain = false;
  private boolean d_loadCurve = false;
  private boolean d_adiabatic = false;
  private boolean d_fricHeat = false;
  private double  d_damping = 0.0;
  private boolean d_viscosity = false;
  private boolean d_convert = false;
  private boolean d_impHeat = false;

  // Two panels for time inputs and variable save inputs
  private MPMFlagInputPanel mpmFlagInputPanel = null;

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public MPMInputsPanel(UintahInputPanel parent) {

    // Initialize local variables
    d_integrator = new String("explicit");
    d_mpmAlgo = 8;
    d_failAlgo = new String("none");
    d_impDynamic = true;
    d_solver = new String("petsc");

    d_gridReset = true;
    d_accStrain = false;
    d_loadCurve = false;
    d_adiabatic = false;
    d_fricHeat = false;
    d_damping = 0.0;
    d_viscosity = false;
    d_convert = false;
    d_impHeat = false;


    // Create the panels
    mpmFlagInputPanel = new MPMFlagInputPanel();

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(mpmFlagInputPanel, gbc);
    add(mpmFlagInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    mpmFlagInputPanel.refresh();
  }

  //------------------------------------------------------------------~~~~~
  // Write out in Uintah format
  //------------------------------------------------------------------~~~~~
  public void writeUintah(PrintWriter pw, String tab) {
   
    if (pw == null) return;
    mpmFlagInputPanel.writeUintah(pw, tab);

  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);

          mpmFlagInputPanel.writeUintah(pw, "  ");

          pw.close();
          fw.close();
        } catch (Exception event) {
          System.out.println("Could not write to file "+outputFile.getName());
        }
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Time input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class MPMFlagInputPanel extends JPanel {

    // Data and components
    private JComboBox integratorComB = null;
    private JComboBox mpmAlgoComB = null;
    private JCheckBox gridResetCB = null;
    private DecimalField minMassEntry = null;
    private DecimalField maxVelEntry = null;

    private JCheckBox accStrainCB = null;
    private JCheckBox loadCurvesCB = null;
    private JCheckBox adiabaticCB = null;
    private JCheckBox fricHeatCB = null;
    private JCheckBox dampingCB = null;
    private DecimalField dampCoeffEntry = null;
    private JCheckBox viscosityCB = null;
    private DecimalField viscCoeff1Entry = null;
    private DecimalField viscCoeff2Entry = null;
    private JComboBox failAlgoComB = null;
    private JCheckBox convertCB = null;
    private JComboBox implicitAlgoComB = null;
    private JComboBox implicitSolverComB = null;
    private JCheckBox impHeatCB = null;
    private DecimalField convDispEntry = null;
    private DecimalField convEnergyEntry = null;
    private IntegerField maxItersDecDeltEntry = null;
    private DecimalField delTDecFacEntry = null;
    private IntegerField minItersIncDeltEntry = null;
    private DecimalField delTIncFacEntry = null;
    private IntegerField maxItersRestartEntry = null;

    private JLabel implicitAlgoLabel = null;
    private JLabel implicitSolverLabel = null;
    private JLabel convDispLabel = null;
    private JLabel convEnergyLabel = null;
    private JLabel maxItersDecDeltLabel = null;
    private JLabel delTDecFacLabel = null;
    private JLabel minItersIncDeltLabel = null;
    private JLabel delTIncFacLabel = null;
    private JLabel maxItersRestartLabel = null;

    public MPMFlagInputPanel() {

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Add the first panel
      JPanel panel1 = new JPanel(new GridLayout(1,0));

      JLabel integratorLabel = new JLabel("Time Integration");
      integratorComB = new JComboBox();
      integratorComB.addItem("Explicit");
      integratorComB.addItem("Implicit");
      panel1.add(integratorLabel); panel1.add(integratorComB); 

      JLabel mpmAlgoLabel = new JLabel("MPM Algorithm");
      mpmAlgoComB = new JComboBox();
      mpmAlgoComB.addItem("Standard");
      mpmAlgoComB.addItem("GIMP");
      panel1.add(mpmAlgoLabel); panel1.add(mpmAlgoComB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               1, 1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1); 

      // Add the second panel
      GridBagLayout gbPanel2 = new GridBagLayout();
      GridBagConstraints gbcPanel2 = new GridBagConstraints();
      JPanel panel2 = new JPanel(gbPanel2);

      JLabel minMassLabel = new JLabel("Minimum Allowed Particle Mass");
      minMassEntry = new DecimalField(1.0e-12, 9, true);
      UintahGui.setConstraints(gbcPanel2, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               1, 1, 5);
      gbPanel2.setConstraints(minMassLabel, gbcPanel2);
      UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 0, 
                               1, 1, 5);
      gbPanel2.setConstraints(minMassEntry, gbcPanel2);
      panel2.add(minMassLabel);  panel2.add(minMassEntry);

      JLabel maxVelLabel = new JLabel("Maximum Allowed Particle Velocity");
      maxVelEntry = new DecimalField(1.0e8, 9, true);
      UintahGui.setConstraints(gbcPanel2, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 
                               1, 1, 5);
      gbPanel2.setConstraints(maxVelLabel, gbcPanel2);
      UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 1, 
                               1, 1, 5);
      gbPanel2.setConstraints(maxVelEntry, gbcPanel2);
      panel2.add(maxVelLabel); panel2.add(maxVelEntry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 2, 
                               1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2); 

      // Add the third panel
      JPanel panel3 = new JPanel(new GridLayout(3,0));

      gridResetCB = new JCheckBox("Reset Grid Every Timestep");
      gridResetCB.setSelected(true);
      panel3.add(gridResetCB);

      accStrainCB = new JCheckBox("Accumulate Strain Energy");
      accStrainCB.setSelected(true);
      panel3.add(accStrainCB);

      loadCurvesCB = new JCheckBox("Use Load Curves");
      loadCurvesCB.setSelected(false);
      panel3.add(loadCurvesCB);

      adiabaticCB = new JCheckBox("Adiabatic Heating In Solids");
      adiabaticCB.setSelected(false);
      panel3.add(adiabaticCB);

      fricHeatCB = new JCheckBox("Heating Due to Contact Friction");
      fricHeatCB.setSelected(false);
      panel3.add(fricHeatCB);

      impHeatCB = new JCheckBox("Implicit Heat Conduction");
      impHeatCB.setSelected(false);
      impHeatCB.setEnabled(false);
      panel3.add(impHeatCB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 3, 
                               1, 1, 5);
      gb.setConstraints(panel3, gbc);
      add(panel3); 

      // Add the fourth panel
      GridBagLayout gbPanel4 = new GridBagLayout();
      GridBagConstraints gbcPanel4 = new GridBagConstraints();
      JPanel panel4 = new JPanel(gbPanel4);

      dampingCB = new JCheckBox("Velocity Damping");
      dampingCB.setSelected(false);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               1, 1, 5);
      gbPanel4.setConstraints(dampingCB, gbcPanel4);
      panel4.add(dampingCB);

      JLabel dampCoeffLabel = new JLabel("Velocity Damping Coeff.");
      dampCoeffEntry = new DecimalField(0.0, 5);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 0, 
                               1, 1, 5);
      gbPanel4.setConstraints(dampCoeffLabel, gbcPanel4);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 2, 0, 
                               1, 1, 5);
      gbPanel4.setConstraints(dampCoeffEntry, gbcPanel4);
      panel4.add(dampCoeffLabel); panel4.add(dampCoeffEntry);

      viscosityCB = new JCheckBox("Force Damping (Artificial Viscosity)");
      viscosityCB.setSelected(false);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 
                               1, 1, 5);
      gbPanel4.setConstraints(viscosityCB, gbcPanel4);
      panel4.add(viscosityCB);

      JLabel viscCoeff1Label = new JLabel("Viscosity Coeff. 1");
      viscCoeff1Entry = new DecimalField(0.2, 5);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 1, 
                               1, 1, 5);
      gbPanel4.setConstraints(viscCoeff1Label, gbcPanel4);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 2, 1, 
                               1, 1, 5);
      gbPanel4.setConstraints(viscCoeff1Entry, gbcPanel4);
      panel4.add(viscCoeff1Label); panel4.add(viscCoeff1Entry);

      JLabel viscCoeff2Label = new JLabel("Viscosity Coeff. 2");
      viscCoeff2Entry = new DecimalField(2.0, 5);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 3, 1, 
                               1, 1, 5);
      gbPanel4.setConstraints(viscCoeff2Label, gbcPanel4);
      UintahGui.setConstraints(gbcPanel4, GridBagConstraints.NONE, 
                               1.0, 1.0, 4, 1, 
                               1, 1, 5);
      gbPanel4.setConstraints(viscCoeff2Entry, gbcPanel4);
      panel4.add(viscCoeff2Label); panel4.add(viscCoeff2Entry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 4, 
                               1, 1, 5);
      gb.setConstraints(panel4, gbc);
      add(panel4); 

      // Add the sixth panel
      JPanel panel6 = new JPanel(new GridLayout(2,0));

      JLabel failAlgoLabel = new JLabel("Failure Algorithm");
      failAlgoComB = new JComboBox();
      failAlgoComB.addItem("No Failure");
      failAlgoComB.addItem("Zero Stress After Failure");
      failAlgoComB.addItem("Allow No Tension After Failure");
      failAlgoComB.addItem("Remove Mass After Failure");
      failAlgoComB.addItem("Keep Stress After Failure");
      panel6.add(failAlgoLabel); panel6.add(failAlgoComB);

      convertCB = new JCheckBox("Create New Material For Failed Particles");
      convertCB.setSelected(false);
      convertCB.setEnabled(false);
      panel6.add(convertCB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 6, 
                               1, 1, 5);
      gb.setConstraints(panel6, gbc);
      add(panel6); 

      // Create the seventh panel
      JPanel panel7 = new JPanel(new GridLayout(1,0));

      implicitAlgoLabel = new JLabel("Implicit MPM Algorithm");
      implicitAlgoComB = new JComboBox();
      implicitAlgoComB.addItem("Dynamic");
      implicitAlgoComB.addItem("Quasistatic");
      panel7.add(implicitAlgoLabel); panel7.add(implicitAlgoComB);

      implicitSolverLabel = new JLabel("Implicit MPM Solver");
      implicitSolverComB = new JComboBox();
      implicitSolverComB.addItem("Petsc Solver");
      implicitSolverComB.addItem("Simple Solver");
      panel7.add(implicitSolverLabel); panel7.add(implicitSolverComB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 7, 
                               1, 1, 5);
      gb.setConstraints(panel7, gbc);
      add(panel7); 

      // Create the ninth panel
      GridBagLayout gbPanel9 = new GridBagLayout();
      GridBagConstraints gbcPanel9 = new GridBagConstraints();
      JPanel panel9 = new JPanel(gbPanel9);

      convDispLabel = new JLabel("Convergence Tol. (Disp.)");
      convDispEntry = new DecimalField(1.0e-10,9,true);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               1, 1, 5);
      gbPanel9.setConstraints(convDispLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 0, 
                               1, 1, 5);
      gbPanel9.setConstraints(convDispEntry, gbcPanel9);
      panel9.add(convDispLabel); panel9.add(convDispEntry);

      convEnergyLabel = new JLabel("Convergence Tol. (Energy)");
      convEnergyEntry = new DecimalField(4.0e-10,9,true);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 2, 0, 
                               1, 1, 5);
      gbPanel9.setConstraints(convEnergyLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 3, 0, 
                               1, 1, 5);
      gbPanel9.setConstraints(convEnergyEntry, gbcPanel9);
      panel9.add(convEnergyLabel); panel9.add(convEnergyEntry);

      maxItersDecDeltLabel = 
        new JLabel("Max. Iter. Before Timestep Decrease ");
      maxItersDecDeltEntry = new IntegerField(12, 5);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 
                               1, 1, 5);
      gbPanel9.setConstraints(maxItersDecDeltLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 1, 
                               1, 1, 5);
      gbPanel9.setConstraints(maxItersDecDeltEntry, gbcPanel9);
      panel9.add(maxItersDecDeltLabel); panel9.add(maxItersDecDeltEntry);

      delTDecFacLabel = new JLabel("Timestep Decrease Factor");
      delTDecFacEntry = new DecimalField(0.5,4);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 2, 1, 
                               1, 1, 5);
      gbPanel9.setConstraints(delTDecFacLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 3, 1, 
                               1, 1, 5);
      gbPanel9.setConstraints(delTDecFacEntry, gbcPanel9);
      panel9.add(delTDecFacLabel); panel9.add(delTDecFacEntry);

      minItersIncDeltLabel = 
        new JLabel("Min. Iter. Before Timestep Increase ");
      minItersIncDeltEntry = new IntegerField(4, 5);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 2, 
                               1, 1, 5);
      gbPanel9.setConstraints(minItersIncDeltLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 2, 
                               1, 1, 5);
      gbPanel9.setConstraints(minItersIncDeltEntry, gbcPanel9);
      panel9.add(minItersIncDeltLabel); panel9.add(minItersIncDeltEntry);

      delTIncFacLabel = new JLabel("Timestep Increase Factor");
      delTIncFacEntry = new DecimalField(2.0,4);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 2, 2, 
                               1, 1, 5);
      gbPanel9.setConstraints(delTIncFacLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 3, 2, 
                               1, 1, 5);
      gbPanel9.setConstraints(delTIncFacEntry, gbcPanel9);
      panel9.add(delTIncFacLabel); panel9.add(delTIncFacEntry);

      maxItersRestartLabel = 
        new JLabel("Max. Iter. Before Timestep Restart ");
      maxItersRestartEntry = new IntegerField(15, 5);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 3, 
                               1, 1, 5);
      gbPanel9.setConstraints(maxItersRestartLabel, gbcPanel9);
      UintahGui.setConstraints(gbcPanel9, GridBagConstraints.NONE, 
                               1.0, 1.0, 1, 3, 
                               1, 1, 5);
      gbPanel9.setConstraints(maxItersRestartEntry, gbcPanel9);
      panel9.add(maxItersRestartLabel); panel9.add(maxItersRestartEntry);

      // Disable the implicit stuff if the integrator is explicit
      implicitAlgoLabel.setEnabled(false);
      implicitSolverLabel.setEnabled(false);
      convDispLabel.setEnabled(false);
      convEnergyLabel.setEnabled(false);
      maxItersDecDeltLabel.setEnabled(false);
      delTDecFacLabel.setEnabled(false);
      minItersIncDeltLabel.setEnabled(false);
      delTIncFacLabel.setEnabled(false);
      maxItersRestartLabel.setEnabled(false);

      implicitAlgoComB.setEnabled(false);
      implicitSolverComB.setEnabled(false);
      impHeatCB.setEnabled(false);
      convDispEntry.setEnabled(false);
      convEnergyEntry.setEnabled(false);
      maxItersDecDeltEntry.setEnabled(false);
      delTDecFacEntry.setEnabled(false);
      minItersIncDeltEntry.setEnabled(false);
      delTIncFacEntry.setEnabled(false);
      maxItersRestartEntry.setEnabled(false);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 9, 
                               1, 1, 5);
      gb.setConstraints(panel9, gbc);
      add(panel9); 

      // Create and add the listeners
      ComboBoxListener comboBoxListener = new ComboBoxListener();
      integratorComB.addItemListener(comboBoxListener);
      mpmAlgoComB.addItemListener(comboBoxListener);
      failAlgoComB.addItemListener(comboBoxListener);
      implicitAlgoComB.addItemListener(comboBoxListener);
      implicitSolverComB.addItemListener(comboBoxListener);

      CheckBoxListener cbListener = new CheckBoxListener();
      gridResetCB.addItemListener(cbListener);
      accStrainCB.addItemListener(cbListener);
      loadCurvesCB.addItemListener(cbListener);
      adiabaticCB.addItemListener(cbListener);
      fricHeatCB.addItemListener(cbListener);
      dampingCB.addItemListener(cbListener);
      viscosityCB.addItemListener(cbListener);
      convertCB.addItemListener(cbListener);
      impHeatCB.addItemListener(cbListener);
     
    }

    //-----------------------------------------------------------------------
    // Refresh
    //-----------------------------------------------------------------------
    public void refresh() {
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : ComboBoxListener
    // Purpose : Listens for item picked in combo box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class ComboBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
        // Get the combo box that has been changed
        Object source = e.getItemSelectable();

        // Get the item that has been selected
        String item = String.valueOf(e.getItem());
          
        if (source == integratorComB) {
          if (item == "Explicit") {
            d_integrator = "explicit";

            implicitAlgoLabel.setEnabled(false);
            implicitSolverLabel.setEnabled(false);
            convDispLabel.setEnabled(false);
            convEnergyLabel.setEnabled(false);
            maxItersDecDeltLabel.setEnabled(false);
            delTDecFacLabel.setEnabled(false);
            minItersIncDeltLabel.setEnabled(false);
            delTIncFacLabel.setEnabled(false);
            maxItersRestartLabel.setEnabled(false);

            implicitAlgoComB.setEnabled(false);
            implicitSolverComB.setEnabled(false);
            impHeatCB.setEnabled(false);
            convDispEntry.setEnabled(false);
            convEnergyEntry.setEnabled(false);
            maxItersDecDeltEntry.setEnabled(false);
            delTDecFacEntry.setEnabled(false);
            minItersIncDeltEntry.setEnabled(false);
            delTIncFacEntry.setEnabled(false);
            maxItersRestartEntry.setEnabled(false);
          } else {
            d_integrator = "implicit";

            implicitAlgoLabel.setEnabled(true);
            implicitSolverLabel.setEnabled(true);
            convDispLabel.setEnabled(true);
            convEnergyLabel.setEnabled(true);
            maxItersDecDeltLabel.setEnabled(true);
            delTDecFacLabel.setEnabled(true);
            minItersIncDeltLabel.setEnabled(true);
            delTIncFacLabel.setEnabled(true);
            maxItersRestartLabel.setEnabled(true);

            implicitAlgoComB.setEnabled(true);
            implicitSolverComB.setEnabled(true);
            impHeatCB.setEnabled(true);
            convDispEntry.setEnabled(true);
            convEnergyEntry.setEnabled(true);
            maxItersDecDeltEntry.setEnabled(true);
            delTDecFacEntry.setEnabled(true);
            minItersIncDeltEntry.setEnabled(true);
            delTIncFacEntry.setEnabled(true);
            maxItersRestartEntry.setEnabled(true);
          }
        } else if (source == mpmAlgoComB) {
          if (item == "Standard") {
            d_mpmAlgo = 8;
          } else {
            d_mpmAlgo = 27;
          }
        } else if (source == failAlgoComB) {
          if (item == "No Failure") {
            d_failAlgo = "none";
          } else if (item == "Remove Mass After Failure") {
            d_failAlgo = "RemoveMass";
            convertCB.setEnabled(true);
          } else if (item == "Zero Stress After Failure") {
            d_failAlgo = "ZeroStress";
            convertCB.setEnabled(true);
          } else if (item == "Allow No Tension After Failure") {
            d_failAlgo = "AllowNoTension";
            convertCB.setEnabled(true);
          } else {
            d_failAlgo = "KeepStress";
            convertCB.setEnabled(true);
          }
        } else if (source == implicitAlgoComB) {
          if (item == "Dynamic") {
            d_impDynamic = true;
          } else {
            d_impDynamic = false;
          }
        } else if (source == implicitSolverComB) {
          if (item == "Petsc Solver") {
            d_solver = "petsc";
          } else {
            d_solver = "simple";
          }
        }
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : CheckBoxListener
    // Purpose : Listens for item seleceted in check box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class CheckBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
        // Get the combo box that has been changed
        Object source = e.getItemSelectable();

        if (source == gridResetCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_gridReset = true;
          } else {
            d_gridReset = false;
          }
        } else if (source == accStrainCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_accStrain = true;
          } else {
            d_accStrain = false;
          }
        } else if (source == loadCurvesCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_loadCurve = true;
          } else {
            d_loadCurve = false;
          }
        } else if (source == adiabaticCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_adiabatic = true;
          } else {
            d_adiabatic = false;
          }
        } else if (source == fricHeatCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_fricHeat = true;
          } else {
            d_fricHeat = false;
          }
        } else if (source == dampingCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            dampCoeffEntry.setEnabled(true);
            d_damping = dampCoeffEntry.getValue();
          } else {
            d_damping = 0.0;
            dampCoeffEntry.setEnabled(false);
          }
        } else if (source == viscosityCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_viscosity = true;
            viscCoeff1Entry.setEnabled(true);
            viscCoeff2Entry.setEnabled(true);
          } else {
            d_viscosity = false;
            viscCoeff1Entry.setEnabled(false);
            viscCoeff2Entry.setEnabled(false);
          }
        } else if (source == convertCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_convert = true;
          } else {
            d_convert = false;
          }
        } else if (source == impHeatCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_impHeat = true;
          } else {
            d_impHeat = false;
          }
        }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw, String tab) {
      
      if (pw == null) return;

      String tab1 = new String(tab+"  ");

      // Write the data
      pw.println(tab+"<MPM>");
      pw.println(tab1+"<time_integrator> "+d_integrator+" </time_integrator>");
      pw.println(tab1+"<nodes8or27> "+d_mpmAlgo+" </nodes8or27>");
      pw.println(tab1+"<minimum_particle_mass> "+minMassEntry.getValue()+
                   " </minimum_particle_mass>");
      pw.println(tab1+"<maximum_particle_velocity> "+maxVelEntry.getValue()+
                   " </maximum_particle_velocity>");
      if (!d_gridReset) {
        pw.println(tab1+"<do_grid_reset> "+d_gridReset+" </do_grid_reset>");
      }
      pw.println(tab1+"<accumulate_strain_energy> "+d_accStrain+
                   " </accumulate_strain_energy>");
      pw.println(tab1+"<use_load_curves> "+d_loadCurve+" </use_load_curves>");
      pw.println(tab1+"<turn_on_adiabatic_heating> "+d_adiabatic+
                   " </turn_on_adiabatic_heating>");
      pw.println(tab1+"<do_contact_friction_heating> "+d_fricHeat+
                   " </do_contact_friction_heating>");
      pw.println(tab1+"<artificial_damping_coeff> "+d_damping+
                   " </artificial_damping_coeff>");
      pw.println(tab1+"<artificial_viscosity> "+d_viscosity+
                   " </artificial_viscosity>");
      pw.println(tab1+"<artificial_viscosity_coeff1> "+
                   viscCoeff1Entry.getValue()+
                   " </artificial_viscosity_coeff1>");
      pw.println(tab1+"<artificial_viscosity_coeff2> "+
                   viscCoeff2Entry.getValue()+
                   " </artificial_viscosity_coeff2>");
      pw.println(tab1+"<erosion algorithm = \""+d_failAlgo+"\"/>");
      pw.println(tab1+"<create_new_particles> "+d_convert+
                   " </create_new_particles>");
      if (d_integrator == "implicit") {
        pw.println(tab1+"<dynamic> "+d_impDynamic+" </dynamic>");
        pw.println(tab1+"<solver> "+d_solver+" </solver>");
        pw.println(tab1+"<DoImplicitHeatConduction> "+d_impHeat+
                     " </DoImplicitHeatConduction>");
        pw.println(tab1+"<convergence_criteria_disp> "+
                     convDispEntry.getValue()+
                     " </convergence_criteria_disp>");
        pw.println(tab1+"<convergence_criteria_energy> "+
                     convEnergyEntry.getValue()+
                     " </convergence_criteria_energy>");
        pw.println(tab1+"<num_iters_to_decrease_delT> "+
                     maxItersDecDeltEntry.getValue()+
                     " </num_iters_to_decrease_delT>");
        pw.println(tab1+"<delT_decrease_factor> "+
                     delTDecFacEntry.getValue()+" </delT_decrease_factor>");
        pw.println(tab1+"<num_iters_to_increase_delT> "+
                     minItersIncDeltEntry.getValue()+
                     " </num_iters_to_increase_delT>");
        pw.println(tab1+"<delT_increase_factor> "+
                     delTIncFacEntry.getValue()+" </delT_increase_factor>");
        pw.println(tab1+"<iters_before_timestep_restart> "+
                     maxItersRestartEntry.getValue()+
                     " </iters_before_timestep_restart>");
      }
      pw.println(tab1+"<min_grid_level> 0 </min_grid_level>");
      pw.println(tab1+"<max_grid_level> 1000 </max_grid_level>");
      pw.println(tab+"</MPM>");
      pw.println(tab);
    }
  }

}
