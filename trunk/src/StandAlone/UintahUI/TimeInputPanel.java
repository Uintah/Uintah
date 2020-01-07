//**************************************************************************
// Class   : TimeInputPanel
// Purpose : A panel that contains widgets to take inputs for
//           the time + timestep information
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;

public class TimeInputPanel extends JPanel 
                            implements ItemListener,
                                       ActionListener {

  // Data and components
  private JTextField titleEntry = null;
  private JComboBox simCompCB = null;

  private DecimalField initTimeEntry = null;
  private DecimalField maxTimeEntry = null;
  private IntegerField maxNofStepsEntry = null;

  private DecimalField deltInitEntry = null;
  private DecimalField deltMinEntry = null;
  private DecimalField deltMaxEntry = null;
  private DecimalField maxDeltIncEntry = null;
  private DecimalField deltMultiplierEntry = null;

  private JTextField udaFilenameEntry = null;
  private DecimalField outputIntervalEntry = null;
  private IntegerField outputTimestepIntervalEntry = null;
  private IntegerField checkPointCycleEntry = null;
  private DecimalField checkPointIntervalEntry = null;
  private IntegerField checkPointTimestepIntervalEntry = null;

  private String d_simType = null;
  private GeneralInputsPanel d_parent = null;
  private boolean d_outputStep = false;
  private boolean d_checkStep = false;

  public TimeInputPanel(String simType,
                        GeneralInputsPanel parent) {

    // Init data
    d_simType = simType;
    d_parent = parent;
    d_outputStep = false;
    d_checkStep = false;

    // Header panel
    JPanel panel1 = new JPanel(new GridLayout(3,0));
    JLabel titleLabel = new JLabel("Simulation Title");
    titleEntry = new JTextField("Test Simulation",20);
    JLabel simCompLabel = new JLabel("Simulation Component");
    simCompCB = new JComboBox();
    simCompCB.addItem("Select one");
    simCompCB.addItem("MPM");
    simCompCB.addItem("ICE");
    simCompCB.addItem("MPMICE");
    JLabel udaFilenameLabel = new JLabel("Output UDA Filename");
    udaFilenameEntry = new JTextField("test.uda",20);
    panel1.add(titleLabel);
    panel1.add(titleEntry);
    panel1.add(simCompLabel);
    panel1.add(simCompCB);
    panel1.add(udaFilenameLabel);
    panel1.add(udaFilenameEntry);

    // Time inputs
    JPanel panel2 = new JPanel(new GridLayout(8,0));
    JLabel initTimeLabel = new JLabel("Initial Time");
    initTimeEntry = new DecimalField(0.0,8,true);
    JLabel maxTimeLabel = new JLabel("Maximum Time");
    maxTimeEntry = new DecimalField(1.0,8,true);
    JLabel maxNofStepsLabel = new JLabel("Maximum Timesteps");
    maxNofStepsEntry = new IntegerField(0, 5);
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
    panel2.add(initTimeLabel);
    panel2.add(initTimeEntry);
    panel2.add(maxTimeLabel);
    panel2.add(maxTimeEntry);
    panel2.add(maxNofStepsLabel);
    panel2.add(maxNofStepsEntry);
    panel2.add(deltInitLabel);
    panel2.add(deltInitEntry);
    panel2.add(deltMinLabel);
    panel2.add(deltMinEntry);
    panel2.add(deltMaxLabel);
    panel2.add(deltMaxEntry);
    panel2.add(maxDeltIncLabel);
    panel2.add(maxDeltIncEntry);
    panel2.add(deltMultiplierLabel);
    panel2.add(deltMultiplierEntry);

    // Output intervals
    JPanel panel4 = new JPanel(new GridLayout(6,0));

    JRadioButton outputIntervalRB = 
      new JRadioButton("Output Time Interval");
    outputIntervalRB.setActionCommand("outputtime");
    outputIntervalRB.setSelected(true);
    outputIntervalEntry = new DecimalField(1.0e-6, 8, true);

    JRadioButton outputTimestepIntervalRB = 
      new JRadioButton("Output Timestep Interval");
    outputTimestepIntervalRB.setActionCommand("outputstep");
    outputTimestepIntervalEntry = new IntegerField(10, 4);

    ButtonGroup outputBG = new ButtonGroup();
    outputBG.add(outputIntervalRB);
    outputBG.add(outputTimestepIntervalRB);

    JLabel checkPointCycleLabel = new JLabel("Check Point Cycle");
    checkPointCycleEntry = new IntegerField(2, 4);

    JRadioButton checkPointIntervalRB = 
      new JRadioButton("Checkpoint Time Interval");
    checkPointIntervalRB.setActionCommand("checktime");
    checkPointIntervalRB.setSelected(true);
    checkPointIntervalEntry = new DecimalField(5.0e-6, 8, true);

    JRadioButton checkPointTimestepIntervalRB = 
      new JRadioButton("Checkpoint Timestep Interval");
    checkPointTimestepIntervalRB.setActionCommand("checkstep");
    checkPointTimestepIntervalEntry = new IntegerField(50, 4);

    ButtonGroup checkPointBG = new ButtonGroup();
    checkPointBG.add(checkPointIntervalRB);
    checkPointBG.add(checkPointTimestepIntervalRB);

    panel4.add(outputIntervalRB);
    panel4.add(outputIntervalEntry);
    panel4.add(outputTimestepIntervalRB);
    panel4.add(outputTimestepIntervalEntry);
    panel4.add(checkPointCycleLabel);
    panel4.add(checkPointCycleEntry);
    panel4.add(checkPointIntervalRB);
    panel4.add(checkPointIntervalEntry);
    panel4.add(checkPointTimestepIntervalRB);
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
    gb.setConstraints(panel4, gbc);
    add(panel4);

    // Create and add the listeners
    simCompCB.addItemListener(this);
    outputIntervalRB.addActionListener(this);
    outputTimestepIntervalRB.addActionListener(this);
    checkPointIntervalRB.addActionListener(this);
    checkPointTimestepIntervalRB.addActionListener(this);
  }

  public void actionPerformed(ActionEvent e) {
    if ((e.getActionCommand()).equals("outputtime")) {
      d_outputStep = false;
    } else if ((e.getActionCommand()).equals("outputstep")) {
      d_outputStep = true;
    } else if ((e.getActionCommand()).equals("checktime")) {
      d_checkStep = false;
    } else if ((e.getActionCommand()).equals("checkstep")) {
      d_checkStep = true;
    }
  }

  //-----------------------------------------------------------------------
  // Purpose : Listens for item picked in combo box and takes action as
  //           required.
  //-----------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());
    if (item.equals(new String("MPM"))) {
        d_simType = "mpm";
    } else if (item.equals(new String("ICE"))) {
        d_simType = "ice";
    } else if (item.equals(new String("MPMICE"))) {
        d_simType = "mpmice";
    } else if (item.equals(new String("RMPMICE"))) {
        d_simType = "rmpmice";
    } else if (item.equals(new String("SMPM"))) {
        d_simType = "smpm";
    } else if (item.equals(new String("SMPMICE"))) {
        d_simType = "smpmice";
    }
    d_parent.updateTabs(d_simType);
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;

    String tab1 = new String(tab+"  ");

    // Write the data
    pw.println(tab+"<Meta>");
    pw.println(tab1+"<title> "+titleEntry.getText()+" </title>");
    pw.println(tab+"</Meta>");
    pw.println(tab);

    pw.println(tab+"<SimulationComponent>");
    pw.println(tab1+"<type>"+d_simType+"</type>");
    pw.println(tab+"</SimulationComponent>");
    pw.println(tab);
           
    pw.println(tab+"<Time>");
    pw.println(tab1+"<initTime> "+ initTimeEntry.getValue()+
                    " </initTime>");
    pw.println(tab1+"<maxTime> "+ maxTimeEntry.getValue()+
                    " </maxTime>");
    if (maxNofStepsEntry.getValue() > 0) {
      pw.println(tab1+"<max_Timesteps> "+ maxNofStepsEntry.getValue()+
                      " </max_Timesteps>");
    }
    pw.println(tab1+"<delt_init> "+ deltInitEntry.getValue()+ " </delt_init>");
    pw.println(tab1+"<delt_min> "+ deltMinEntry.getValue()+ " </delt_min>");
    pw.println(tab1+"<delt_max> "+ deltMaxEntry.getValue()+ " </delt_max>");
    pw.println(tab1+"<max_delt_increase> "+ maxDeltIncEntry.getValue()+
                    " </max_delt_increase>");
    pw.println(tab1+"<timestep_multiplier> "+ deltMultiplierEntry.getValue()+
                    " </timestep_multiplier>");
    pw.println(tab+"</Time>");
    pw.println(tab);

    pw.println(tab+"<DataArchiver>");
    pw.println(tab1+"<filebase> "+ udaFilenameEntry.getText()+
                    " </filebase>");
    if (d_outputStep) {
      pw.println(tab1+"<outputTimestepInterval> "+
                 outputTimestepIntervalEntry.getValue()+
                 " </outputTimestepInterval>");
    } else {
      pw.println(tab1+"<outputInterval> "+ outputIntervalEntry.getValue()+
                      " </outputInterval>");
    }
    if (d_checkStep) {
      pw.println(tab1+"<checkpoint cycle=\""+ checkPointCycleEntry.getValue()+
                 "\" timestepInterval=\""+
                 checkPointTimestepIntervalEntry.getValue()+
                 "\"/>");
    } else {
      pw.println(tab1+"<checkpoint cycle=\""+ checkPointCycleEntry.getValue()+
                 "\" interval=\""+ checkPointIntervalEntry.getValue()+
                 "\"/>");
    }
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
  }

}
