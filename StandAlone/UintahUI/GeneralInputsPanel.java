/**************************************************************************
// Program : GeneralInputsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           1) The title 
//           2) The simulation component
//           3) The time + timestep information
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.event.*;
import java.text.DecimalFormat;
import java.util.Vector;

//**************************************************************************
// Class   : GeneralInputsPanel
// Purpose : The following components are realized in the panel :
//             1) A gridbag layout that contains the components.
//             2) Text fields to take the inputs.
//             3) A button that saves the input data.
//**************************************************************************
public class GeneralInputsPanel extends JPanel {

  // Static variables

  // Data
  private UintahInputPanel d_parent = null;

  // Two panels for time inputs and variable save inputs
  private TimeInputPanel timeInputPanel = null;
  private VariableSaveInputPanel saveInputPanel = null;

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public GeneralInputsPanel(UintahInputPanel parent) {

    // Initialize local variables
    d_parent = parent;

    // Create the panels
    timeInputPanel = new TimeInputPanel();
    saveInputPanel = new VariableSaveInputPanel();

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
    gb.setConstraints(timeInputPanel, gbc);
    add(timeInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 1,0, 1,1, 5);
    gb.setConstraints(saveInputPanel, gbc);
    add(saveInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
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

          timeInputPanel.writeUintah(pw);
          saveInputPanel.writeUintah(pw);

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
  private class TimeInputPanel extends JPanel {

    // Data and components
    private JTextField titleEntry = null;
    private JComboBox simCompCB = null;

    private DecimalField initTimeEntry = null;
    private DecimalField maxTimeEntry = null;
    private WholeNumberField maxNofStepsEntry = null;

    private DecimalField deltInitEntry = null;
    private DecimalField deltMinEntry = null;
    private DecimalField deltMaxEntry = null;
    private DecimalField maxDeltIncEntry = null;
    private DecimalField deltMultiplierEntry = null;

    private JTextField udaFilenameEntry = null;
    private DecimalField outputIntervalEntry = null;
    private WholeNumberField outputTimestepIntervalEntry = null;
    private WholeNumberField checkPointCycleEntry = null;
    private DecimalField checkPointIntervalEntry = null;
    private WholeNumberField checkPointTimestepIntervalEntry = null;

    private String d_simType = null;

    public TimeInputPanel() {

      // Init data
      d_simType = new String("mpm");

      // Create the panels
      JPanel panel1 = new JPanel(new GridLayout(2,0));
      JPanel panel2 = new JPanel(new GridLayout(3,0));
      JPanel panel3 = new JPanel(new GridLayout(5,0));
      JPanel panel4 = new JPanel(new GridLayout(6,0));
 
      // Create the components
      JLabel titleLabel = new JLabel("Simulation Title");
      titleEntry = new JTextField("Test Simulation",20);
      JLabel simCompLabel = new JLabel("Simulation Component");
      simCompCB = new JComboBox();
      simCompCB.addItem("MPM");
      simCompCB.addItem("ICE");
      simCompCB.addItem("MPMICE");
      simCompCB.addItem("RMPMICE");
      simCompCB.addItem("SMPM");
      simCompCB.addItem("SMPMICE");
      panel1.add(titleLabel);
      panel1.add(titleEntry);
      panel1.add(simCompLabel);
      panel1.add(simCompCB);

      JLabel initTimeLabel = new JLabel("Initial Time");
      initTimeEntry = new DecimalField(0.0,8,true);
      JLabel maxTimeLabel = new JLabel("Maximum Time");
      maxTimeEntry = new DecimalField(1.0,8,true);
      JLabel maxNofStepsLabel = new JLabel("Maximum Timesteps");
      maxNofStepsEntry = new WholeNumberField(1000, 5);
      panel2.add(initTimeLabel);
      panel2.add(initTimeEntry);
      panel2.add(maxTimeLabel);
      panel2.add(maxTimeEntry);
      panel2.add(maxNofStepsLabel);
      panel2.add(maxNofStepsEntry);

      JLabel deltInitLabel = new JLabel("Initial Timestep Size");
      deltInitEntry = new DecimalField(1.0e-9, 8, true);
      JLabel deltMinLabel = new JLabel("Minimum Timestep Size");
      deltMinEntry = new DecimalField(0.0, 8, true);
      JLabel deltMaxLabel = new JLabel("Maximum Timestep Size");
      deltMaxEntry = new DecimalField(1.0e-3, 8, true);
      JLabel maxDeltIncLabel = new JLabel("Maximum Timestep Increase Factor");
      maxDeltIncEntry = new DecimalField(1.0, 6);
      JLabel deltMultiplierLabel = new JLabel("Timestep Multiplier");
      deltMultiplierEntry = new DecimalField(0.5, 6);
      panel3.add(deltInitLabel);
      panel3.add(deltInitEntry);
      panel3.add(deltMinLabel);
      panel3.add(deltMinEntry);
      panel3.add(deltMaxLabel);
      panel3.add(deltMaxEntry);
      panel3.add(maxDeltIncLabel);
      panel3.add(maxDeltIncEntry);
      panel3.add(deltMultiplierLabel);
      panel3.add(deltMultiplierEntry);

      JLabel udaFilenameLabel = new JLabel("Output UDA Filename");
      udaFilenameEntry = new JTextField("test.uda",20);
      JLabel outputIntervalLabel = new JLabel("Output Time Interval");
      outputIntervalEntry = new DecimalField(1.0e-6, 8, true);
      JLabel outputTimestepIntLabel = new JLabel("Output Timestep Interval");
      outputTimestepIntervalEntry = new WholeNumberField(10, 4);
      JLabel checkPointCycleLabel = new JLabel("Check Point Cycle");
      checkPointCycleEntry = new WholeNumberField(2, 4);
      JLabel checkPointIntervalLabel = new JLabel("Checkpoint Time Interval");
      checkPointIntervalEntry = new DecimalField(5.0e-6, 8, true);
      JLabel checkPointTimestepIntLabel = 
        new JLabel("Checkpoint Timestep Interval");
      checkPointTimestepIntervalEntry = new WholeNumberField(50, 4);
      panel4.add(udaFilenameLabel);
      panel4.add(udaFilenameEntry);
      panel4.add(outputIntervalLabel);
      panel4.add(outputIntervalEntry);
      panel4.add(outputTimestepIntLabel);
      panel4.add(outputTimestepIntervalEntry);
      panel4.add(checkPointCycleLabel);
      panel4.add(checkPointCycleEntry);
      panel4.add(checkPointIntervalLabel);
      panel4.add(checkPointIntervalEntry);
      panel4.add(checkPointTimestepIntLabel);
      panel4.add(checkPointTimestepIntervalEntry);

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Grid bag layout
      UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                      1.0, 1.0, 0,0, 1,1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1);

      UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                      1.0, 1.0, 0,1, 1,1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2);

      UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                      1.0, 1.0, 0,2, 1,1, 5);
      gb.setConstraints(panel3, gbc);
      add(panel3);

      UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                      1.0, 1.0, 0,3, 1,1, 5);
      gb.setConstraints(panel4, gbc);
      add(panel4);

      // Create and add the listeners
      ComboBoxListener comboBoxListener = new ComboBoxListener();
      simCompCB.addItemListener(comboBoxListener);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : ComboBoxListener
    // Purpose : Listens for item picked in combo box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class ComboBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {

        // Get the item that has been selected
        String item = String.valueOf(e.getItem());
        if (item == "MPM") {
          d_simType = "mpm";
        } else if (item == "ICE") {
          d_simType = "ice";
        } else if (item == "MPMICE") {
          d_simType = "mpmice";
        } else if (item == "RMPMICE") {
          d_simType = "rmpmice";
        } else if (item == "SMPM") {
          d_simType = "smpm";
        } else if (item == "SMPMICE") {
          d_simType = "smpmice";
        }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw) {
      
      if (pw == null) return;


        // Write the data
        pw.println("  <Meta>");
        pw.println("    <title>"+titleEntry.getText()+"</title>");
        pw.println("  </Meta>");
        pw.println(" ");

        pw.println("  <SimulationComponent>");
        pw.println("    <type>"+d_simType+"</type>");
        pw.println("  </SimulationComponent>");
        pw.println(" ");
           
        pw.println("  <Time>");
        pw.println("    <initTime>"+ initTimeEntry.getValue()+
                           "</initTime>");
        pw.println("    <maxTime>"+ maxTimeEntry.getValue()+
                           "</maxTime>");
        pw.println("    <max_iterations>"+ maxNofStepsEntry.getValue()+
                           "</max_iterations>");

        pw.println("    <delt_init>"+ deltInitEntry.getValue()+ 
                           "</delt_int>");
        pw.println("    <delt_min>"+ deltMinEntry.getValue()+
                           "</delt_min>");
        pw.println("    <delt_max>"+ deltMaxEntry.getValue()+
                           "</delt_max>");
        pw.println("    <max_delt_increase>"+
                           maxDeltIncEntry.getValue()+
                           "</max_delt_increase>");
        pw.println("    <timestep_multiplier>"+
                           deltMultiplierEntry.getValue()+
                           "</timestep_multiplier>");
        pw.println("  </Time>");
        pw.println(" ");

        pw.println("  <DataArchiver>");
        pw.println("    <filebase>"+
                           udaFilenameEntry.getText()+
                           "</filebase>");
        pw.println("    <outputInterval>"+
                           outputIntervalEntry.getValue()+
                           "</outputInterval>");
        pw.println("    <outputTimestepInterval>"+
                           outputTimestepIntervalEntry.getValue()+
                           "<outputTimestepInterval>");
        pw.println("    <checkpoint cycle=\""+
                           checkPointCycleEntry.getValue()+
                           "\" interval=\""+
                           checkPointIntervalEntry.getValue()+
                           "\"/>");
        pw.println("    <checkpoint cycle=\""+
                           checkPointCycleEntry.getValue()+
                           "\" timstepInterval=\""+
                           checkPointTimestepIntervalEntry.getValue()+
                           "\"/>");
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Variable input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class VariableSaveInputPanel extends JPanel 
                                       implements ItemListener {

    // Data 
    public Vector d_summedVar = null;
    public Vector d_partVar = null;
    public Vector d_gridVar = null;
    public Vector d_cellVar = null;

    public Vector d_summedVarStr = null;
    public Vector d_partVarStr = null;
    public Vector d_gridVarStr = null;
    public Vector d_cellVarStr = null;

    public Vector d_summedVarState = null;
    public Vector d_partVarState = null;
    public Vector d_gridVarState = null;
    public Vector d_cellVarState = null;

    // Components
    private JCheckBox kineticEnergyCB = null;
    private JCheckBox accStrainEnergyCB = null;
    private JCheckBox strainEnergyCB = null;
    private JCheckBox momentumCB = null;
    private JCheckBox totalMassCB = null;
    private JCheckBox centerOfMassCB = null;

    private JCheckBox p_particleIDCB = null;
    private JCheckBox p_positionCB = null;
    private JCheckBox p_massCB = null;
    private JCheckBox p_volumeCB = null;
    private JCheckBox p_temperatureCB = null;
    private JCheckBox p_stressCB = null;
    private JCheckBox p_deformationGradientCB = null;
    private JCheckBox p_displacementCB = null;
    private JCheckBox p_velocityCB = null;
    private JCheckBox p_externalForceCB = null;
    private JCheckBox p_localizedCB = null;
    private JCheckBox p_damageCB = null;
    private JCheckBox p_porosityCB = null;
    private JCheckBox p_plasticStrainCB = null;
    private JCheckBox p_plasticStrainRateCB = null;
    private JCheckBox p_strainRateCB = null;

    private JCheckBox g_massCB = null;
    private JCheckBox g_volumeCB = null;
    private JCheckBox g_velocityCB = null;
    private JCheckBox g_stressCB = null;
    private JCheckBox g_accelerationCB = null;

    private JCheckBox cc_densityCB = null;
    private JCheckBox cc_temperatureCB = null;
    private JCheckBox cc_velocityCB = null;
    private JCheckBox cc_spVolumeCB = null;
    private JCheckBox cc_volFracCB = null;
    private JCheckBox cc_pressureCB = null;
    private JCheckBox cc_equilPressureCB = null;
    private JCheckBox cc_intEnergyLCB = null;
    private JCheckBox cc_intEnergySourceCB = null;
    private JCheckBox cc_TdotCB = null;
    private JCheckBox cc_momentumLCB = null;
    private JCheckBox cc_momentumSourceCB = null;
    private JCheckBox cc_delPDilatateCB = null;

    public VariableSaveInputPanel() {

      // Initialize vector 
      d_summedVar = new Vector();
      d_partVar = new Vector();
      d_gridVar = new Vector();
      d_cellVar = new Vector();

      d_summedVarStr = new Vector();
      d_partVarStr = new Vector();
      d_gridVarStr = new Vector();
      d_cellVarStr = new Vector();

      d_summedVarState = new Vector();
      d_partVarState = new Vector();
      d_gridVarState = new Vector();
      d_cellVarState = new Vector();

      // Create panels for these check boxes
      JPanel panel0 = new JPanel(new GridLayout(1,0));
      JPanel panel1 = new JPanel(new GridLayout(0,1));
      JPanel panel2 = new JPanel(new GridLayout(0,1));
      JPanel panel3 = new JPanel(new GridLayout(0,1));
      JPanel panel4 = new JPanel(new GridLayout(0,1));

      JLabel label0 = new JLabel("Variables to be saved");
      panel0.add(label0);

      JLabel label1 = new JLabel("Summed Variables");
      panel1.add(label1);

      kineticEnergyCB = new JCheckBox("Kinetic Energy");
      strainEnergyCB = new JCheckBox("Inc. Strain Energy");
      accStrainEnergyCB = new JCheckBox("Total Strain Energy");
      momentumCB = new JCheckBox("Momentum");
      totalMassCB = new JCheckBox("Mass");
      centerOfMassCB = new JCheckBox("Center of Mass");

      d_summedVar.addElement(kineticEnergyCB);
      d_summedVarStr.addElement(new String("KineticEnergy"));
      d_summedVar.addElement(strainEnergyCB);
      d_summedVarStr.addElement(new String("StrainEnergy"));
      d_summedVar.addElement(accStrainEnergyCB);
      d_summedVarStr.addElement(new String("AccStrainEnergy"));
      d_summedVar.addElement(momentumCB);
      d_summedVarStr.addElement(new String("CenterOfMassVelocity"));
      d_summedVar.addElement(totalMassCB);
      d_summedVarStr.addElement(new String("TotalMass"));
      d_summedVar.addElement(centerOfMassCB);
      d_summedVarStr.addElement(new String("CenterOfMassPosition"));

      int numSummedVar = d_summedVar.size();
      for (int ii = 0; ii < numSummedVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_summedVar.elementAt(ii);
        checkBox.setSelected(true);
        checkBox.addItemListener(this);
        d_summedVarState.addElement(new Boolean(true));
        panel1.add(checkBox);
      }

      JLabel label2 = new JLabel("Particle Variables");
      panel2.add(label2);

      p_particleIDCB = new JCheckBox("Particle ID");
      p_positionCB = new JCheckBox("Position");
      p_massCB = new JCheckBox("Mass");
      p_volumeCB = new JCheckBox("Volume");
      p_temperatureCB = new JCheckBox("Temperature");
      p_deformationGradientCB = new JCheckBox("Def. Gradient");
      p_displacementCB = new JCheckBox("Displacement");
      p_velocityCB = new JCheckBox("Velocity");
      p_strainRateCB = new JCheckBox("Strain Rate");
      p_externalForceCB = new JCheckBox("External Force");
      p_stressCB = new JCheckBox("Stress");
      p_localizedCB = new JCheckBox("Failed Particles");
      p_damageCB = new JCheckBox("Damage");
      p_porosityCB = new JCheckBox("Porosity");
      p_plasticStrainCB = new JCheckBox("Plastic Strain");
      p_plasticStrainRateCB = new JCheckBox("Plastic Strain Rate");

      d_partVar.addElement(p_particleIDCB);
      d_partVarStr.addElement(new String("p.particleID"));
      d_partVar.addElement(p_positionCB);
      d_partVarStr.addElement(new String("p.x"));
      d_partVar.addElement(p_massCB);
      d_partVarStr.addElement(new String("p.mass"));
      d_partVar.addElement(p_volumeCB);
      d_partVarStr.addElement(new String("p.volume"));
      d_partVar.addElement(p_temperatureCB);
      d_partVarStr.addElement(new String("p.temperature"));
      d_partVar.addElement(p_deformationGradientCB);
      d_partVarStr.addElement(new String("p.deformationMeasure"));
      d_partVar.addElement(p_displacementCB);
      d_partVarStr.addElement(new String("p.displacement"));
      d_partVar.addElement(p_velocityCB);
      d_partVarStr.addElement(new String("p.velocity"));
      d_partVar.addElement(p_strainRateCB);
      d_partVarStr.addElement(new String("p.strainRate"));
      d_partVar.addElement(p_externalForceCB);
      d_partVarStr.addElement(new String("p.externalforce"));
      d_partVar.addElement(p_stressCB);
      d_partVarStr.addElement(new String("p.stress"));
      d_partVar.addElement(p_localizedCB);
      d_partVarStr.addElement(new String("p.localized"));
      d_partVar.addElement(p_damageCB);
      d_partVarStr.addElement(new String("p.damage"));
      d_partVar.addElement(p_porosityCB);
      d_partVarStr.addElement(new String("p.porosity"));
      d_partVar.addElement(p_plasticStrainCB);
      d_partVarStr.addElement(new String("p.plasticStrain"));
      d_partVar.addElement(p_plasticStrainRateCB);
      d_partVarStr.addElement(new String("p.plasticStrainRate"));

      int numPartVar = d_partVar.size();
      for (int ii = 0; ii < numPartVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_partVar.elementAt(ii);
        checkBox.setSelected(true);
        checkBox.addItemListener(this);
        d_partVarState.addElement(new Boolean(true));
        panel2.add(checkBox);
      }

      JLabel label3 = new JLabel("Grid Variables");;
      panel3.add(label3);

      g_massCB = new JCheckBox("Mass");
      g_volumeCB = new JCheckBox("Volume");
      g_velocityCB = new JCheckBox("Velocity");
      g_stressCB = new JCheckBox("Stress");
      g_accelerationCB = new JCheckBox("Acceleration");

      d_gridVar.addElement(g_massCB);
      d_gridVarStr.addElement(new String("g.mass"));
      d_gridVar.addElement(g_volumeCB);
      d_gridVarStr.addElement(new String("g.volume"));
      d_gridVar.addElement(g_velocityCB);
      d_gridVarStr.addElement(new String("g.velocity"));
      d_gridVar.addElement(g_stressCB);
      d_gridVarStr.addElement(new String("g.stressFS"));
      d_gridVar.addElement(g_accelerationCB);
      d_gridVarStr.addElement(new String("g.acceleration"));

      int numGridVar = d_gridVar.size();
      for (int ii = 0; ii < numGridVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_gridVar.elementAt(ii);
        checkBox.setSelected(true);
        checkBox.addItemListener(this);
        d_gridVarState.addElement(new Boolean(true));
        panel3.add(checkBox);
      }

      JLabel label4 = new JLabel("Cell-Centered Variables");
      panel4.add(label4);

      cc_densityCB = new JCheckBox("Density");
      cc_temperatureCB = new JCheckBox("Temperature");
      cc_velocityCB = new JCheckBox("Velocity");
      cc_spVolumeCB = new JCheckBox("Specific Volume");
      cc_volFracCB = new JCheckBox("Volume Fraction");
      cc_pressureCB = new JCheckBox("Pressure");
      cc_equilPressureCB = new JCheckBox("Equilibriation Pressure");
      cc_intEnergyLCB = new JCheckBox("Internal Energy");
      cc_intEnergySourceCB = new JCheckBox("Internal Energy Source");
      cc_TdotCB = new JCheckBox("Temperature Rate");
      cc_momentumLCB = new JCheckBox("Momentum");
      cc_momentumSourceCB = new JCheckBox("Momentum Source");
      cc_delPDilatateCB = new JCheckBox("delP Dilatation");

      d_cellVar.addElement(cc_densityCB);
      d_cellVarStr.addElement(new String("rho_CC"));
      d_cellVar.addElement(cc_temperatureCB);
      d_cellVarStr.addElement(new String("temp_CC"));
      d_cellVar.addElement(cc_velocityCB);
      d_cellVarStr.addElement(new String("vel_CC"));
      d_cellVar.addElement(cc_spVolumeCB);
      d_cellVarStr.addElement(new String("sp_vol_CC"));
      d_cellVar.addElement(cc_volFracCB);
      d_cellVarStr.addElement(new String("vol_frac_CC"));
      d_cellVar.addElement(cc_pressureCB);
      d_cellVarStr.addElement(new String("press_CC"));
      d_cellVar.addElement(cc_equilPressureCB);
      d_cellVarStr.addElement(new String("press_equil_CC"));
      d_cellVar.addElement(cc_intEnergyLCB);
      d_cellVarStr.addElement(new String("int_eng_L_CC"));
      d_cellVar.addElement(cc_intEnergySourceCB);
      d_cellVarStr.addElement(new String("intE_source_CC"));
      d_cellVar.addElement(cc_TdotCB);
      d_cellVarStr.addElement(new String("Tdot"));
      d_cellVar.addElement(cc_momentumLCB);
      d_cellVarStr.addElement(new String("mom_L_CC"));
      d_cellVar.addElement(cc_momentumSourceCB);
      d_cellVarStr.addElement(new String("mom_source_CC"));
      d_cellVar.addElement(cc_delPDilatateCB);
      d_cellVarStr.addElement(new String("delP_Dilatate"));

      int numCellVar = d_cellVar.size();
      for (int ii = 0; ii < numCellVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_cellVar.elementAt(ii);
        checkBox.setSelected(true);
        checkBox.addItemListener(this);
        d_cellVarState.addElement(new Boolean(true));
        panel4.add(checkBox);
      }

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Grid bag layout
      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 
                               GridBagConstraints.REMAINDER,1, 5);
      gb.setConstraints(panel0, gbc);
      add(panel0);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 2, 1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 1, 1, 1, 1, 5);
      gb.setConstraints(panel3, gbc);
      add(panel3);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 1, 2, 1, 1, 5);
      gb.setConstraints(panel4, gbc);
      add(panel4);
    }

    //--------------------------------------------------------------------
    /** Listen for changed item state */
    //--------------------------------------------------------------------
    public void itemStateChanged(ItemEvent e) {

      // Find the check box which has changed
      Object source = e.getItemSelectable();

      // Find which button was selected and toggle the state
      int numSummedVar = d_summedVar.size();
      for (int ii = 0; ii < numSummedVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_summedVar.elementAt(ii);
        if (source == checkBox) {
          if (e.getStateChange() == ItemEvent.DESELECTED) {
            d_summedVarState.setElementAt(new Boolean(false), ii);
          } else {
            d_summedVarState.setElementAt(new Boolean(true), ii);
          }
          return;
        }
      }

      int numPartVar = d_partVar.size();
      for (int ii = 0; ii < numPartVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_partVar.elementAt(ii);
        if (source == checkBox) {
          if (e.getStateChange() == ItemEvent.DESELECTED) {
            d_partVarState.setElementAt(new Boolean(false), ii);
          } else {
            d_partVarState.setElementAt(new Boolean(true), ii);
          }
          return;
        }
      }

      int numGridVar = d_gridVar.size();
      for (int ii = 0; ii < numGridVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_gridVar.elementAt(ii);
        if (source == checkBox) {
          if (e.getStateChange() == ItemEvent.DESELECTED) {
            d_gridVarState.setElementAt(new Boolean(false), ii);
          } else {
            d_gridVarState.setElementAt(new Boolean(true), ii);
          }
          return;
        }
      }

      int numCellVar = d_cellVar.size();
      for (int ii = 0; ii < numCellVar; ++ii) {
        JCheckBox checkBox = (JCheckBox) d_cellVar.elementAt(ii);
        if (source == checkBox) {
          if (e.getStateChange() == ItemEvent.DESELECTED) {
            d_cellVarState.setElementAt(new Boolean(false), ii);
          } else {
            d_cellVarState.setElementAt(new Boolean(true), ii);
          }
          return;
        }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw) {
      
      if (pw == null) return;

      try {

        // Write the data

        int numSummedVar = d_summedVar.size();
        for (int ii = 0; ii < numSummedVar; ++ii) {
          boolean state = 
              ((Boolean) d_summedVarState.elementAt(ii)).booleanValue();
          if (state) {
            String label = (String) d_summedVarStr.elementAt(ii);
            pw.println("      <save label=\""+label+"\"/>");
          }
        }
           
        int numPartVar = d_partVar.size();
        for (int ii = 0; ii < numPartVar; ++ii) {
          boolean state = 
              ((Boolean) d_partVarState.elementAt(ii)).booleanValue();
          if (state) {
            String label = (String) d_partVarStr.elementAt(ii);
            pw.println("      <save label=\""+label+"\"/>");
          }
        }
           
        int numGridVar = d_gridVar.size();
        for (int ii = 0; ii < numGridVar; ++ii) {
          boolean state = 
              ((Boolean) d_gridVarState.elementAt(ii)).booleanValue();
          if (state) {
            String label = (String) d_gridVarStr.elementAt(ii);
            pw.println("      <save label=\""+label+"\"/>");
          }
        }
           
        int numCellVar = d_cellVar.size();
        for (int ii = 0; ii < numCellVar; ++ii) {
          boolean state = 
              ((Boolean) d_cellVarState.elementAt(ii)).booleanValue();
          if (state) {
            String label = (String) d_cellVarStr.elementAt(ii);
            pw.println("      <save label=\""+label+"\"/>");
          }
        }

        pw.println("  </DataArchiver>");
      } catch (Exception e) {
        System.out.println("Could not write to file ");
      }

    }
  }
}
