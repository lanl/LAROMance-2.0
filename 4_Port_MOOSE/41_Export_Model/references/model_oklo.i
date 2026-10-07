


# endtime = 1e-4
endtime = 5.4e7

[GlobalParams]
  order = FIRST
  family = LAGRANGE
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1
  ny = 1
  nz = 1
  xmin = 0.0
  xmax = 1e-4
  ymin = 0.0
  ymax = 1e-4
  zmin = 0.0
  zmax = 1e-4
  elem_type = HEX8
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]


[AuxVariables]
  [temperature]
    order = CONSTANT
    family = MONOMIAL             # Elemental Aux Variable "MONOMIAL"
  []

  [axial]
    order = CONSTANT
    family = MONOMIAL
  []
  [axial_strain]
    order = CONSTANT
    family = MONOMIAL
  []
  [creeprate]
    order = CONSTANT
    family = MONOMIAL
  []
[]


[Functions]
  [temperature_func]
    type = PiecewiseLinear
    # data_file = "./temperature3.csv"
    # format = columns

    x = '0 ${endtime}'
    y = '800.0 800.0'
  []
  [stress_xx_func]
    # type = PiecewiseLinear
    # data_file = "./stress_xx.csv"
    # x_title = time
    # y_title = stress_xx
    # scale_factor = 1e6
    # format = columns

    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '1.0e6 14.0e6'
  []
  [stress_yy_func]
    # type = PiecewiseLinear
    # data_file = "./stress_yy.csv"
    # x_title = time
    # y_title = stress_yy
    # scale_factor = 1e6
    # format = columns

    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '1.0e6 7.0e6'
  []

  [stress_zz_func]
    # type = PiecewiseLinear
    # data_file = "./stress_yy.csv"
    # x_title = time
    # y_title = stress_yy
    # format = columns
    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '0 0.0'
  []
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = finite
    add_variables = true
    incremental = true
    generate_output = 'vonmises_stress  stress_xx stress_yy stress_zz strain_yy strain_xx strain_zz creep_strain_xx creep_strain_yy'
    use_automatic_differentiation = false
    decomposition_method = EigenSolution
    # eigenstrain_names = 'cladding_thermal_eigenstrain'
  []
[]

[AuxKernels]
  [creeprate]
    type = MaterialRealAux
    property = creep_rate
    execute_on = timestep_end
    variable = creeprate
  []
  [temperature]
    variable = temperature
    type = FunctionAux
    function = temperature_func
  []
[]

[BCs]
  # Fix X displacement on left face (prevent X translation)
  [fix_x]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = "left"
  []
  # Fix Y displacement on bottom face (prevent Y translation)
  [fix_y]
    type = DirichletBC
    variable = disp_y
    value = 0
    boundary = "bottom"
  []
  # Fix Z displacement on back face (prevent Z translation)
  [fix_z]
    type = DirichletBC
    variable = disp_z
    value = 0
    boundary = "back"
  []

  # [Pressure]
  [stress_xx_bc]
    type = FunctionNeumannBC
    variable = disp_x
    function = stress_xx_func
    boundary = right
  []
  [stress_yy_bc]
    type = FunctionNeumannBC
    variable = disp_y
    function = stress_yy_func
    boundary = top
  []
  [stress_zz_bc]
    type = FunctionNeumannBC
    variable = disp_z
    function = stress_zz_func
    boundary = front
  []
[]
# []

[Materials]
  [clad_elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 188e9
    poissons_ratio = 0.3
  []
  [clad_stress]
    type = ComputeMultipleInelasticStress
    # tangent_operator = nonlinear
    inelastic_models = 'clad_HT9creep'
    max_iterations = 100000
    relative_tolerance = 1e-8
    absolute_tolerance = 1e-10
  []
  # [clad_HT9creep]
  #   type = HT9CreepUpdate
  #   temperature = temperature
  #   outputs = all
  #   primary_creep_model = MFH
  #   secondary_creep_model = MFH
  #   tertiary_creep_model = MFH
  #   irradiation_creep_model = MFH
  #   use_effective_time_for_tertiary = true
  #   use_effective_time_for_primary = true
  #   fast_neutron_flux = 9.9e-07
  # []
  [clad_HT9creep]
    type = testmodel
    temperature = temperature
    use_kdtree_indexing = true
    
    initial_cell_dislocation_density = 1.0e14
    initial_wall_dislocation_density = 8.0e13
    environmental_factor = 1.2e-9

    max_relative_wall_dislocation_increment = 0.9
    max_relative_cell_dislocation_increment = 0.9

    outputs = all
    max_inelastic_increment = 1e-1

    cell_input_window_low_failure = USELIMIT
    wall_input_window_low_failure = USELIMIT
    cell_input_window_high_failure = USELIMIT
    wall_input_window_high_failure = USELIMIT
    old_strain_input_window_high_failure = ERROR
    stress_input_window_low_failure = USELIMIT
    temperature_input_window_low_failure = USELIMIT
    verbose = false
  []

[]

[Dampers]
  [limitY]
    type = MaxIncrement
    max_increment = 5e-3 #5e-6
    variable = disp_y
  []
[]

[Preconditioning]
  [SMP]
    type = SMP
    full = TRUE
  []
[]

[Executioner]
  type = Transient
  solve_type = 'newton'
  petsc_options = '-snes_ksp_ew'
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu       superlu_dist'
  line_search = 'bt'

  l_max_its = 1000
  nl_max_its = 1000
  l_tol = 1e-4
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10
  automatic_scaling = true
  compute_scaling_once = false
  start_time = 0.0
  end_time = ${endtime}
  dt = 1.0e-4
  dtmin = 1e-8
  dtmax = 1e7

  [TimeStepper]
    type = IterationAdaptiveDT
    dt = 1.0
    timestep_limiting_postprocessor = material_timestep
    growth_factor = 1.5
    cutback_factor = 0.3
  []
[]

[Postprocessors]
  [_dt] # time step
    type = TimestepSize
  []
  [material_timestep]
    type = MaterialTimeStepPostprocessor
    outputs = console
  []
  [max_dispx]
    type = NodalExtremeValue
    value_type = max
    variable = disp_x
    boundary = 'top'
    execute_on = timestep_end
  []
  [max_dispy]
    type = NodalExtremeValue
    value_type = max
    variable = disp_y
    boundary = 'top'
    execute_on = timestep_end
  []
  [max_hoop_stress]
    type = ElementExtremeValue
    value_type = max
    variable = stress_xx
    execute_on = timestep_end
  []
  [max_hoop_strain]
    type = ElementExtremeValue
    value_type = max
    variable = strain_xx
    execute_on = timestep_end
  []
  [max_hoop_creep_strain]
    type = ElementExtremeValue
    value_type = max
    variable = creep_strain_xx
    execute_on = timestep_end
  []
  [max_axial_strain]
    type = ElementExtremeValue
    value_type = max
    variable = strain_yy
    execute_on = timestep_end
  []
  [max_radial_strain]
    type = ElementExtremeValue
    value_type = max
    variable = strain_zz
    execute_on = timestep_end
  []
  [max_creep_rate]
    type = ElementExtremeValue
    value_type = max
    variable = creeprate
    execute_on = timestep_end
  []
  [vonmises_stress]
    type = ElementAverageValue
    variable = vonmises_stress
    execute_on = timestep_end
  []
  [max_stress_xx]
    type = ElementAverageValue
    variable = stress_xx
    execute_on = timestep_end
  []
  [max_stress_yy]
    type = ElementAverageValue
    variable = stress_yy
    execute_on = timestep_end
  []
  [max_stress_zz]
    type = ElementAverageValue
    variable = stress_zz
    execute_on = timestep_end
  []
  [temp]
    type = FunctionValuePostprocessor
    function = temperature_func
  []
  [wall_dislocations]
    type = ElementAverageValue
    variable = wall_dislocations
  [../]
  [cell_dislocations]
    type = ElementAverageValue
    variable = cell_dislocations
  [../]

[]

[Outputs]
  [csv]
    type = CSV
  []
  exodus = true
  perf_graph = true

  [console]
    type = Console
    max_rows = 5
    execute_on = 'timestep_end'
  []
[]
