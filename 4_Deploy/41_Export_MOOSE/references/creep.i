# some global input variables
endtime = 1e8 #  ~ 3 years
P = 200.0e6
T = 2000.0
rhoc = 1.0e+14
dt = 1e-6
model = TestModel

[Mesh]
  [rod]
    type = GeneratedMeshGenerator
    dim = 3
    xmin = 0.0
    xmax = 1.0e-4
    ymin = 0.0
    ymax = 1.0e-4
    zmin = 0.0
    zmax = 1.0e-4
    nx = 1
    ny = 1
    nz = 1
    elem_type = HEX8
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[AuxVariables]
  [./temperature]
  order = CONSTANT
  family = MONOMIAL             # Elemental Aux Variable "MONOMIAL"
  [../]
[]

[AuxKernels]
  [./temperature]
    variable = temperature
    type = FunctionAux
    function = temp_func
  [../]
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
  [pressure_x]
    type = FunctionNeumannBC
    variable = disp_x
    boundary = right
    function = pres_prof_pos
  []
[]

[Functions]
  [./pres_prof_pos]
    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '${P} ${P}'
  [../]
  [./temp_func]
    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '${T} ${T}'
  [../]
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 180.0e9
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = 'rom_stress_prediction_base'
  []
  [rom_stress_prediction_base]
    type = ${model} # <-- model comes from the global params set at the top of this input file.
    # Input to the ROM stress prediction model
    temperature = temperature
    initial_cell_dislocation_density = ${rhoc} 
    # Outputs
    outputs = all
    # Update limits for coupled inputs that are computed from output rates via time-step multiplication (e.g., inelastic strain increment and cell dislocation density increment)
    max_inelastic_increment = 1e-2 
    max_relative_cell_dislocation_increment = 1e-1 
    #  Out-of-bounds handling (USELIMIT > snap to bounds, EXTRAPOLATE > use extrapolated value, ERROR > throw an error, WARN > issue a warning, IGNORE/DONOTHING > ignore the violation)
    stress_input_window_low_failure = EXTRAPOLATE
    stress_input_window_high_failure = USELIMIT
    temperature_input_window_high_failure = ERROR
    temperature_input_window_low_failure = USELIMIT
    cell_input_window_low_failure = USELIMIT
    cell_input_window_high_failure = USELIMIT
    # Extra
    internal_solve_output_on = on_error
    internal_solve_full_iteration_history = false
    use_kdtree_indexing = true # <- speeds up surrogate element searches: true unless mesh is highly non-uniform. False behavior will cause significant slow-down.
    verbose = false
  []
[]

[Preconditioning]
  [./smp]
    type = SMP # FDP finite difference preconditioning, SMP analytic Jacobian
    full = true
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu       superlu_dist'
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10
  line_search = 'none'

  automatic_scaling = true
  compute_scaling_once = true
  use_pre_SMO_residual = true
  l_max_its = 1
  nl_max_its = 10
  end_time = ${endtime}
  nl_forced_its = 1

  [./TimeStepper]
    timestep_limiting_postprocessor = time_step_limit
    type = IterationAdaptiveDT
    optimal_iterations = 5
    iteration_window = 2
    dt = ${dt}
    growth_factor = 1.5
    cutback_factor = 0.3
  []
[]

[Postprocessors]
  [./finding_element_duration]
    type = ElementAverageMaterialProperty
    mat_prop = duration_element_search
    execute_on = 'timestep_end'
  [../]
  [./brute_force_fallback]
    type = ElementAverageMaterialProperty
    mat_prop = fallback_brute_force
    execute_on = 'timestep_end'
  [../]
  
  [./_dt]
    type = TimestepSize     # size of the time step per increment
  [../]
  [./nliterations]            # Number of iterations needed to converge timestep
    type = NumNonlinearIterations
  [../]
  [./literations]            # Number of linear iterations needed to converge timestep
    type = NumLinearIterations
  [../]
  [./numreseval]             # Number of residual evaluations
    type =  NumResidualEvaluations
  [../]
  [./residual]              # force residual after convergence (per increment)
    type = Residual
    residual_type =  final
  [../]
  [./initial_residual_before]  # force residual at initial guess (per increment) before preset?
    type = Residual
    residual_type = initial_before_preset
  [../]
  [./initial_residual_after] # force residual at initial guess (per increment) after preset?
    type = Residual
    residual_type = initial_after_preset
  [../]
  [./active_time]           # Time computer spent on simulation
    type = PerfGraphData
    section_name = "Root"
    data_type = total
  [../]
  [./creep_timestep]
    type = MaterialTimeStepPostprocessor
  [../]
  [creep_rate]
    type = ElementAverageMaterialProperty
    mat_prop = creep_rate
  []
  [./effective_strain_avg]
    type = ElementAverageValue
    variable = effective_creep_strain
  [../]
  [./vonmises_stress]
    type = ElementAverageValue
    variable = vonmises_stress
  [../]
  [./temperature]
    type = ElementAverageValue
    variable = temperature
  [../]
  [./cell_dislocations]
    type = ElementAverageValue
    variable = cell_dislocations
  [../]

  [./strain_xx]
    type = ElementAverageValue
    variable = strain_xx
  [../]
  [./strain_yy]
    type = ElementAverageValue
    variable = strain_yy
  [../]
  [./strain_zz]
    type = ElementAverageValue
    variable = strain_zz
  [../]

  [./stress_zz]
    type = ElementAverageValue
    variable = stress_zz
  [../]
  [./stress_xx]
    type = ElementAverageValue
    variable = stress_xx
  [../]
  [./stress_yy]
    type = ElementAverageValue
    variable = stress_yy
  [../]


  [./stress_xx_max]
    type = ElementExtremeValue
    variable = stress_xx
  [../]
  [./stress_yy_max]
    type = ElementExtremeValue
    variable = stress_yy
  [../]

  # extras
  [./time_step_limit]
    type = MaterialTimeStepPostprocessor
  [../]
[]

[Outputs]
  # file_base = ${model}_1-element_T-${T}_P-${P}_rhoc-${rhoc}
  csv = true
  gnuplot = true
  print_linear_residuals = true
  perf_graph = true
  output_on = 'initial timestep_end'
[]
