[Mesh]
  type = GeneratedMesh
  dim = 3
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[AuxVariables]
  [temperature]
  []
[]

[AuxKernels]
  [temp_aux]
    type = FunctionAux
    variable = temperature
    function = temperature_fcn
    execute_on = 'initial timestep_begin'
  []
[]

[Functions]
  [vmJ2_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    
    x_index_in_file = 0
    y_index_in_file = 1
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [temperature_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 2
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [evm_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 3
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [rhoc_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 4
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [rhow_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 5
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [defect_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 6
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [rhoc_rate_soln_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 8
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [rhow_rate_soln_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 9
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
  [creep_rate_soln_fcn]
    type = PiecewiseConstant
    data_file = benchmarks.csv
    x_index_in_file = 0
    y_index_in_file = 10
    format = columns
    xy_in_file_only = false
    direction = LEFT_INCLUSIVE
  []
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [all]
        strain = FINITE
        add_variables = true
        volumetric_locking_correction = true
        generate_output = 'vonmises_stress'
      []
    []
  []
[]


[BCs]
  [symmx]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [symmy]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [symmz]
    type = DirichletBC
    variable = disp_z
    boundary = back
    value = 0
  []
  [pull_x]
    type = DirichletBC
    variable = disp_x
    boundary = right
    value = 1e-5 # This is required to make a non-zero effective trial stress so radial return is engaged
  []
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 200e9
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = rom_stress_prediction
  []
  [defect_rate]
    type = GenericFunctionMaterial
    prop_names = 'defect_rate'
    prop_values = 'defect_rate_fcn'
    outputs = all
  []
  [rom_stress_prediction]
    type = test
    temperature = temperature
    effective_inelastic_strain_name = effective_creep_strain
    internal_solve_full_iteration_history = true
    apply_strain = false
    verbose = false
    outputs = all
    effective_stress_forcing_function = vmJ2_fcn
    wall_dislocation_density_forcing_function = rhoc_fcn
    cell_dislocation_density_forcing_function = rhom_fcn
    old_creep_strain_forcing_function = evm_fcn
    environmental_factor = defect_rate
    wall_input_window_low_failure = IGNORE
    wall_input_window_high_failure = IGNORE
    cell_input_window_low_failure = IGNORE
    cell_input_window_high_failure = IGNORE
    temperature_input_window_low_failure = ERROR
    temperature_input_window_high_failure = ERROR
    stress_input_window_low_failure = ERROR
    stress_input_window_high_failure = ERROR
    old_strain_input_window_low_failure = ERROR
    old_strain_input_window_high_failure = ERROR
    environment_input_window_low_failure = ERROR
    environment_input_window_high_failure = ERROR
    
  []
[]

[Executioner]
  type = Transient

  solve_type = 'NEWTON'

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  nl_abs_tol = 1e-1 # Nothing is really being solved here, so loose tolerances are okay

  dt = 0.1
  end_time = 98.6
  # end_time = 0.2
  timestep_tolerance = 1e-3
[]

[Postprocessors]
  [old_strain_in]
    type = FunctionValuePostprocessor
    function = evm_fcn
    execute_on = 'TIMESTEP_END initial'
    outputs = console
  []
  [temperature]
    type = ElementAverageValue
    variable = temperature
    # outputs = console
  []
  [rhoc_rate]
    type = ElementAverageMaterialProperty
    mat_prop = cell_dislocation_rate
  []
  [rhow_rate]
    type = ElementAverageMaterialProperty
    mat_prop = wall_dislocation_rate
  []
  [creep_rate]
    type = ElementAverageMaterialProperty
    mat_prop = creep_rate
  []
  [rhoc_in]
    type = FunctionValuePostprocessor
    function = rhom_fcn
    execute_on = 'TIMESTEP_END initial'
    # outputs = console
  []
  [rhow_in]
    type = FunctionValuePostprocessor
    function = rhoi_fcn
    execute_on = 'TIMESTEP_END initial'
    # outputs = console
  []
  [vmJ2]
    type = FunctionValuePostprocessor
    function = vmJ2_fcn
    execute_on = 'TIMESTEP_END initial'
    # outputs = console
  []
  [defect_rate]
    type = ElementAverageValue
    variable = defect_rate
    # outputs = console
  []
  [rhoc_rate_soln]
    type = FunctionValuePostprocessor
    function = rhom_rate_soln_fcn
    # outputs = console
  []
  [rhow_rate_soln]
    type = FunctionValuePostprocessor
    function = rhoi_rate_soln_fcn
    # outputs = console
  []
  [creep_rate_soln]
    type = FunctionValuePostprocessor
    function = creep_rate_soln_fcn
    # outputs = console
  []

  [rhoc_rate_diff]
    type = ParsedPostprocessor
    pp_names = 'rhoc_rate_soln rhoc_rate'
    expression = '(rhoc_rate_soln - rhoc_rate) / rhoc_rate_soln'
    # outputs = console
  []
  [rhow_rate_diff]
    type = ParsedPostprocessor
    pp_names = 'rhow_rate_soln rhow_rate'
    expression = '(rhow_rate_soln - rhow_rate) / rhow_rate_soln'
    # outputs = console
  []
  [creep_rate_diff]
    type = ParsedPostprocessor
    pp_names = 'creep_rate creep_rate_soln'
    expression = '(creep_rate_soln - creep_rate) / creep_rate_soln'
    # outputs = console
  []

  [z_rhoc_rate_max_diff]
    type = TimeExtremeValue
    postprocessor = rhoc_rate_diff
    value_type = abs_max
  []
  [z_rhow_rate_max_diff]
    type = TimeExtremeValue
    postprocessor = rhow_rate_diff
    value_type = abs_max
  []
  [z_creep_rate_max_diff]
    type = TimeExtremeValue
    postprocessor = creep_rate_diff
    value_type = abs_max
  []
[]

[Outputs]
  csv = true
  execute_on = 'INITIAL TIMESTEP_END FINAL'
[]
